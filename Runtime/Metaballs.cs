using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Profiling;

namespace Tars.Metaballs
{
    [Serializable]
    public struct Metaball
    {
        public float3 position;
        public float maxRadius;
        public float scale;

        public Metaball(float3 position, float maxRadius, float scale)
        {
            this.position = position;
            this.maxRadius = maxRadius;
            this.scale = scale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly float GetIsoValue(in float3 point)
        {
            var dSqrd = math.distancesq(point, position);
            var rSqrd = maxRadius * maxRadius;

            // NOTE: This branch is really bad, can't vectorize...
            // Consider switching to other metaball function. 
            if (dSqrd > rSqrd)
                return 0f;

            var a = (dSqrd / rSqrd) - 1;
            var isoValue = scale * a * a * a * a;
            return isoValue;
        }
    }

    public struct MetaballsJobHandle
    {
        private Metaballs.TesselateJob _tesselateJob;
        private Metaballs.BVHJob _bvhJob;
        private JobHandle _jobHandle;
        private Mesh _mesh;
        private bool _scheduled;

        public bool IsScheduled => _scheduled;
        public bool IsCompleted => _jobHandle.IsCompleted;

        public MetaballsJobHandle(Metaballs.TesselateJob tesselateJob, Metaballs.BVHJob bvhJob, JobHandle jobHandle, Mesh mesh)
        {
            _tesselateJob = tesselateJob;
            _bvhJob = bvhJob;
            _jobHandle = jobHandle;
            _mesh = mesh;
            _scheduled = true;
        }

        public void Complete() => _jobHandle.Complete();

        public void Apply()
        {
            if (!_jobHandle.IsCompleted)
                throw new Exception("Metaballs job is running. Can't apply to mesh until job is completed.");
            if (!_scheduled)
                throw new Exception("Metaballs job is not scheduled. Can't apply.");

            Metaballs.Apply(_mesh, _tesselateJob);
            Metaballs.Dispose(_tesselateJob, _bvhJob);
        }
    }

    public static class Metaballs
    {
        public static readonly ProfilerMarker _triangulate = new ProfilerMarker("FastBlobs.Triangulate");
        public static readonly ProfilerMarker _findSurface = new ProfilerMarker("FastBlobs.FindSurface");

        private static readonly NativeArray<int3> _faces;
        private static readonly NativeArray<int2> _indexToFaceStartEnd;

        public static void Run(NativeArray<Metaball> blobs, float isoLevel, Mesh mesh, int initialCapacity = 1024)
        {
            if (isoLevel < Mathf.Epsilon)
                throw new ArgumentOutOfRangeException();

            var bvhJob = CreateBVHJob(blobs);
            var tesselateJob = CreateTesselateJob(blobs, bvhJob, isoLevel, initialCapacity);

            bvhJob.Run();
            tesselateJob.Run();

            Apply(mesh, tesselateJob);
            Dispose(tesselateJob, bvhJob);
        }

        public static MetaballsJobHandle Schedule(NativeArray<Metaball> blobs, float isoLevel, Mesh mesh, int initialCapacity = 1024)
        {
            if (isoLevel < Mathf.Epsilon)
                throw new ArgumentOutOfRangeException();

            var bvhJob = CreateBVHJob(blobs);
            var tesselateJob = CreateTesselateJob(blobs, bvhJob, isoLevel, initialCapacity);

            var bvhHandle = bvhJob.Schedule();
            var tesselateHandle = tesselateJob.Schedule(bvhHandle);

            return new MetaballsJobHandle(tesselateJob, bvhJob, tesselateHandle, mesh);
        }

        public static void Apply(Mesh mesh, TesselateJob tesselateJob)
        {
            mesh.SetVertexBufferParams(
                tesselateJob.vertices.Length,
                new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3),
                new VertexAttributeDescriptor(VertexAttribute.Normal, VertexAttributeFormat.Float32, 3)
            );
            mesh.SetVertexBufferData<Vertex>(tesselateJob.vertices.AsArray(), 0, 0, tesselateJob.vertices.Length);
            mesh.SetIndexBufferParams(tesselateJob.indices.Length, IndexFormat.UInt32);
            mesh.SetIndexBufferData<uint>(tesselateJob.indices.AsArray(), 0, 0, tesselateJob.indices.Length);
            mesh.subMeshCount = 1;
            mesh.SetSubMesh(0, new SubMeshDescriptor(0, tesselateJob.indices.Length));

            // NOTE: Using BVH tree root AABB as mesh bounds. It is virtually free.
            // But produced mesh bounds do not represent actual mesh bounds (they are slightly larger).
            // Consider replacing this with Mesh.RecalculateBounds() if this is the issue.
            mesh.bounds = new Bounds(
                (Vector3)(float3)(tesselateJob.bvhTree.root.aabb.min + tesselateJob.bvhTree.root.aabb.max) / 2.0f,
                (Vector3)(tesselateJob.bvhTree.root.aabb.max - (float3)tesselateJob.bvhTree.root.aabb.min));
        }

        public static void Dispose(TesselateJob tesselateJob, BVHJob bVHJob)
        {
            bVHJob.nodeContents.Dispose();
            tesselateJob.bvhTree.Dispose();
            tesselateJob.indices.Dispose();
            tesselateJob.vertices.Dispose();
            tesselateJob.visitedCubes.Dispose();
            tesselateJob.sampleLattice.Dispose();
            tesselateJob.orbitToVertices.Dispose();
        }

        private static BVHJob CreateBVHJob(NativeArray<Metaball> blobs)
        {
            var nodes = new NativeList<NativeBVHNode>(Allocator.TempJob);
            var nodeContents = new NativeList<int>(Allocator.TempJob);
            var blobContents = new NativeList<Metaball>(Allocator.TempJob);

            return new BVHJob()
            {
                blobs = blobs,
                nodes = nodes,
                nodeContents = nodeContents,
                blobContents = blobContents
            };
        }

        private static TesselateJob CreateTesselateJob(NativeArray<Metaball> blobs, BVHJob bvhJob, float isoLevel, int initialCapacity)
        {
            var visitedCubes = new NativeHashSet<int3>(initialCapacity, Allocator.TempJob);
            var vertices = new NativeList<Vertex>(Allocator.TempJob);
            var indices = new NativeList<uint>(Allocator.TempJob);
            var sampleLattice = new NativeHashMap<int3, float>(initialCapacity, Allocator.TempJob);
            var orbitToVertices = new NativeHashMap<int3, uint>(initialCapacity, Allocator.TempJob);
            var nativeTree = new NativeBVHTree()
            {
                nodes = bvhJob.nodes,
                blobContents = bvhJob.blobContents
            };

            return new TesselateJob()
            {
                blobs = blobs,
                isoLevel = isoLevel,
                visitedCubes = visitedCubes,
                orbitToVertices = orbitToVertices,
                findSurface = _findSurface,
                triangulate = _triangulate,
                vertices = vertices,
                indices = indices,
                sampleLattice = sampleLattice,
                faces = _faces,
                indexToFaceStartEnd = _indexToFaceStartEnd,
                bvhTree = nativeTree
            };
        }

        // Build a lookup table that allows us to know which faces of a cube are crossed by the surface
        // depending on the cube configuration.
        static Metaballs()
        {
            var managedFaces = new List<int3>(256 * 6);
            var managedStartEnd = new List<int2>(256 * 2);

            // Indices of cube vertices forming a face.
            var faces = new int4[6]
            {
                new int4(1, 2, 5, 6), // + x
                new int4(0, 3, 4, 7), // - x
                new int4(4, 5, 6, 7), // + y
                new int4(0, 1, 2, 3), // - y
                new int4(2, 3, 6, 7), // + z
                new int4(0, 1, 4, 5)  // - z 
            };

            // Normal direction of cube faces.
            var faceOffsets = new int3[6]
            {
                new int3(1, 0, 0),
                new int3(-1, 0, 0),
                new int3(0, 1, 0),
                new int3(0, -1, 0),
                new int3(0, 0, 1),
                new int3(0, 0, -1),
            };

            for (int cubeIndex = 0; cubeIndex < 255; cubeIndex++)
            {
                var startIndex = managedFaces.Count;

                for (int i = 0; i < 6; i++)
                {
                    // Each bool correspond to a vertex in the face of current cube configuraion
                    // And tells us whether it's inside the surface or not.
                    var face = new bool4(
                        ((cubeIndex >> faces[i][0]) & 1) != 0,
                        ((cubeIndex >> faces[i][1]) & 1) != 0,
                        ((cubeIndex >> faces[i][2]) & 1) != 0,
                        ((cubeIndex >> faces[i][3]) & 1) != 0);

                    if (math.any(face) && !math.all(face))
                        managedFaces.Add(faceOffsets[i]);
                }

                managedStartEnd.Add(new int2(startIndex, managedFaces.Count - startIndex));
            }

            _faces = new NativeArray<int3>(managedFaces.ToArray(), Allocator.Persistent);
            _indexToFaceStartEnd = new NativeArray<int2>(managedStartEnd.ToArray(), Allocator.Persistent);
        }

        public struct AABB
        {
            public int3 min;
            public int3 max;
            public int splitAxis;

            public int3 size => max - min;
        }

        public struct NativeBVHTree : IDisposable
        {
            public NativeList<NativeBVHNode> nodes;
            public NativeList<Metaball> blobContents;

            public NativeBVHNode root => nodes[nodes.Length - 1];

            public void Dispose()
            {
                nodes.Dispose();
                blobContents.Dispose();
            }
        }

        public struct NativeBVHNode
        {
            public NativeBVHNode(AABB aabb, int contentsStart, int contentsCount, int leftChildIndex, int rightChildIndex, NativeList<int> contents, NativeList<Metaball> blobContents, NativeArray<Metaball> blobs)
            {
                this.aabb = aabb;
                this.contentsStart = contentsStart;
                this.contentsCount = contentsCount;
                this.leftChildIndex = leftChildIndex;
                this.rightChildIndex = rightChildIndex;
                blobsStart = -1;
                blobsCount = -1;

                if (leftChildIndex == -1)
                {
                    blobsStart = blobContents.Length;
                    for (int i = contentsStart; i < contentsStart + contentsCount; i++)
                        blobContents.Add(blobs[contents[i]]);
                    blobsCount = blobContents.Length - blobsStart;
                }
            }

            public AABB aabb;
            public int contentsStart;
            public int contentsCount;
            public int leftChildIndex;
            public int rightChildIndex;
            public int blobsStart;
            public int blobsCount;
        }

        // TODO: Needs a rewrite for readability. Although perfomance could be improved, too.
        // Also, can theoretically cause stack overflow. Switch from recursion to iterative algorithm. 
        [BurstCompile]
        public struct BVHJob : IJob
        {
            [ReadOnly] public NativeArray<Metaball> blobs;

            public NativeList<NativeBVHNode> nodes;
            public NativeList<int> nodeContents;
            public NativeList<Metaball> blobContents;

            public void Execute()
            {
                var boxes = new NativeArray<AABB>(blobs.Length, Allocator.Temp);
                var rootBox = new AABB();

                for (int i = 0; i < blobs.Length; i++)
                {
                    boxes[i] = new AABB()
                    {
                        min = (int3)(blobs[i].position - new float3(blobs[i].maxRadius)),
                        max = (int3)(blobs[i].position + new float3(blobs[i].maxRadius))
                    };

                    rootBox.min = math.min(rootBox.min, boxes[i].min);
                    rootBox.max = math.max(rootBox.max, boxes[i].max);

                    nodeContents.Add(i);
                }

                var rootIndices = Split((rootBox, boxes.Length), boxes);
                nodes.Add(new NativeBVHNode(rootBox, 0, boxes.Length, rootIndices.indexOfLeft, rootIndices.indexOfRight, nodeContents, blobContents, blobs));
                boxes.Dispose();
            }

            private (int indexOfLeft, int indexOfRight) Split((AABB aabb, int contentsCount) nodeData, NativeArray<AABB> boxes)
            {
                if (nodeData.contentsCount <= 2)
                    return (-1, -1);

                var splitAxis = 0;
                var aabb = nodeData.aabb;

                // TODO: Using (suboptimal) simple split along the center of the longest axis.
                // Consider implementing Surface Area Heuristics to find node split position.
                if (aabb.size.y > aabb.size.x && aabb.size.y > aabb.size.z)
                    splitAxis = 1;
                else if (aabb.size.z > aabb.size.x && aabb.size.z > aabb.size.y)
                    splitAxis = 2;

                var leftChildMax = aabb.max;
                var rightChildMin = aabb.min;

                leftChildMax[splitAxis] -= aabb.size[splitAxis] / 2;
                rightChildMin[splitAxis] = leftChildMax[splitAxis];

                var leftBox = new AABB()
                {
                    min = aabb.min,
                    max = leftChildMax,
                    splitAxis = splitAxis
                };
                var rightBox = new AABB()
                {
                    min = rightChildMin,
                    max = aabb.max,
                    splitAxis = splitAxis
                };

                var leftBoxes = FilterPrimitives(boxes, leftBox);
                var rightBoxes = FilterPrimitives(boxes, rightBox);
                var leftIndex = -1;
                var rightIndex = -1;

                if (leftBoxes.count != nodeData.contentsCount && rightBoxes.count != nodeData.contentsCount)
                {
                    SplitProceed(ref leftIndex, leftBox, leftBoxes.startIndex, leftBoxes.count, boxes);
                    SplitProceed(ref rightIndex, rightBox, rightBoxes.startIndex, rightBoxes.count, boxes);
                }

                return (leftIndex, rightIndex);
            }

            private void SplitProceed(ref int index, AABB box, int startIndex, int count, NativeArray<AABB> boxes)
            {
                var indices = Split((box, count), boxes);
                var leftNode = new NativeBVHNode(box, startIndex, count, indices.indexOfLeft, indices.indexOfRight, nodeContents, blobContents, blobs);
                index = nodes.Length;
                nodes.Add(leftNode);
            }

            private (int startIndex, int count) FilterPrimitives(NativeArray<AABB> boxes, AABB aabb)
            {
                var start = nodeContents.Length;

                for (int i = 0; i < boxes.Length; i++)
                {
                    if (IntersectsWith(boxes[i], aabb))
                        nodeContents.Add(i);
                }

                return (start, nodeContents.Length - start);

                bool IntersectsWith(AABB one, AABB other)
                {
                    return one.min.x <= other.max.x &&
                    one.max.x >= other.min.x &&
                    one.min.y <= other.max.y &&
                    one.max.y >= other.min.y &&
                    one.min.z <= other.max.z &&
                    one.max.z >= other.min.z;
                }
            }
        }

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        public struct Vertex
        {
            public float3 position;
            public float3 normal;
        }

        [BurstCompile]
        public struct TesselateJob : IJob
        {
            [ReadOnly] public float isoLevel;
            [ReadOnly] public NativeArray<Metaball> blobs;
            [ReadOnly] public NativeBVHTree bvhTree;

            public NativeHashSet<int3> visitedCubes;
            public NativeHashMap<int3, uint> orbitToVertices;
            public NativeHashMap<int3, float> sampleLattice;
            public NativeList<Vertex> vertices;
            public NativeList<uint> indices;

            [ReadOnly] private static readonly int[] triangulation;
            [ReadOnly] private static readonly int3[] cubeCorners;
            [ReadOnly] private static readonly int3x2[] edges;
            [ReadOnly] public NativeArray<int3> faces;
            [ReadOnly] public NativeArray<int2> indexToFaceStartEnd;

            private uint _verticeCount;

            public ProfilerMarker triangulate;
            public ProfilerMarker findSurface;

            public unsafe void Execute()
            {
                findSurface.Begin();

                var toTriangulate = new NativeList<(int3 position, byte cubeIndex)>(Allocator.Temp);
                var cubes = new NativeQueue<int3>(Allocator.Temp);

                foreach (var blob in blobs)
                {
                    var point = math.round(blob.position);
                    var node = FindNode((int3)point);

                    while (Sample(in point, (int3)point, in node) > isoLevel)
                    {
                        point += math.right();
                        node = FindNode((int3)point);
                    }

                    var surfaceCube = (int3)(point - math.right());
                    if (visitedCubes.Add(surfaceCube))
                        cubes.Enqueue(surfaceCube);

                    var currentCube = stackalloc float[8];

                    while (cubes.TryDequeue(out var cube))
                    {
                        var cubeNode = FindNode(cube);
                        SampleCube(in cubeNode, in cube, currentCube);
                        var cubeIndex = GetCubeIndex(currentCube);
                        var interval = indexToFaceStartEnd[cubeIndex];

                        for (int i = interval[0]; i < interval[0] + interval[1]; i++)
                            if (visitedCubes.Add(cube + faces[i]))
                                cubes.Enqueue(cube + faces[i]);

                        toTriangulate.Add((cube, cubeIndex));
                    }

                    // NOTE: Can break; here if all metaballs are connected.
                }

                findSurface.End();
                triangulate.Begin();

                vertices.SetCapacity(toTriangulate.Length);
                indices.SetCapacity(toTriangulate.Length * 3);

                foreach (var cube in toTriangulate)
                {
                    BuildCubeMesh(cube.Item1, cube.Item2);
                }

                triangulate.End();

                toTriangulate.Dispose();
                cubes.Dispose();
            }

            private unsafe byte GetCubeIndex(float* cube)
            {
                int index = 0;
                for (int i = 0; i < 8; i++)
                    if (cube[i] > isoLevel)
                        index |= 1 << i;
                return (byte)index;
            }

            private unsafe void SampleCube(in NativeBVHNode node, in int3 position, float* cube)
            {
                for (int i = 0; i < 8; i++)
                {
                    var pos = position + cubeCorners[i];
                    var posFloat = (float3)pos;
                    cube[i] = Sample(in posFloat, in pos, in node);
                }
            }

            private float Sample(in float3 position, in int3 latticePosition, in NativeBVHNode node)
            {
                if (sampleLattice.TryGetValue(latticePosition, out var sampleValue))
                    return sampleValue;

                var sum = 0f;

                for (int i = node.blobsStart; i < node.blobsStart + node.blobsCount; i++)
                    sum += bvhTree.blobContents[i].GetIsoValue(in position);

                sampleLattice.Add(latticePosition, sum);
                return sum;
            }

            // TODO: Should rewrite to allow auto vectorization.
            private NativeBVHNode FindNode(in int3 point)
            {
                var node = bvhTree.root;

                while (true)
                {
                    if (node.leftChildIndex == -1)
                        return node;

                    var leftNode = bvhTree.nodes[node.leftChildIndex];

                    if (leftNode.contentsCount == 0)
                        return node;

                    node = (math.all(point < leftNode.aabb.max)) ? leftNode : bvhTree.nodes[node.rightChildIndex];
                }
            }

            private void BuildCubeMesh(in int3 pos, in byte cubeIndex)
            {
                for (int i = cubeIndex * 16; triangulation[i] != -1; i += 3)
                {
                    var i0 = triangulation[i];
                    var i1 = triangulation[i + 1];
                    var i2 = triangulation[i + 2];

                    var v1 = CreateVertice(in pos, in edges[i0]);
                    var v2 = CreateVertice(in pos, in edges[i1]);
                    var v3 = CreateVertice(in pos, in edges[i2]);

                    indices.Add(TryAddVertice(in v1, (int3)math.round(v1)));
                    indices.Add(TryAddVertice(in v2, (int3)math.round(v2)));
                    indices.Add(TryAddVertice(in v3, (int3)math.round(v3)));
                }
            }

            private float3 CreateVertice(in int3 pos, in int3x2 edge)
            {
                var cornerAPos = edge.c0 + pos;
                var cornerBPos = edge.c1 + pos;
                var cornerAIso = sampleLattice[cornerAPos];
                var cornerBIso = sampleLattice[cornerBPos];

                var t = (isoLevel - cornerAIso) / (cornerBIso - cornerAIso);
                return math.lerp((float3)cornerAPos, (float3)cornerBPos, t);
            }

            private uint TryAddVertice(in float3 vertex, in int3 corner)
            {
                if (orbitToVertices.TryGetValue(corner, out var verticeIndex))
                    return verticeIndex;
                orbitToVertices.Add(corner, _verticeCount);

                var vert = new Vertex()
                {
                    position = vertex,
                    normal = CreateNormal(in vertex, in corner)
                };

                vertices.Add(vert);
                _verticeCount++;
                return _verticeCount - 1;
            }

            private float3 CreateNormal(in float3 vertex, in int3 corner)
            {
                var leaf = FindNode(corner);
                var dirSum = new float3();

                for (int i = leaf.blobsStart; i < leaf.blobsStart + leaf.blobsCount; i++)
                {
                    var blob = bvhTree.blobContents[i];
                    var weight = blob.GetIsoValue(in vertex) / isoLevel;
                    dirSum += math.normalize(vertex - blob.position) * weight;
                }

                return dirSum;
            }

            static TesselateJob()
            {
                // Edge index to positions of vertices forming cube's edge.
                edges = new int3x2[]
                {
                    new int3x2(new int3(0, 0, 0), new int3(1, 0, 0)),
                    new int3x2(new int3(1, 0, 0), new int3(1, 0, 1)),
                    new int3x2(new int3(1, 0, 1), new int3(0, 0, 1)),
                    new int3x2(new int3(0, 0, 1), new int3(0, 0, 0)),
                    new int3x2(new int3(0, 1, 0), new int3(1, 1, 0)),
                    new int3x2(new int3(1, 1, 0), new int3(1, 1, 1)),
                    new int3x2(new int3(1, 1, 1), new int3(0, 1, 1)),
                    new int3x2(new int3(0, 1, 1), new int3(0, 1, 0)),
                    new int3x2(new int3(0, 0, 0), new int3(0, 1, 0)),
                    new int3x2(new int3(1, 0, 0), new int3(1, 1, 0)),
                    new int3x2(new int3(1, 0, 1), new int3(1, 1, 1)),
                    new int3x2(new int3(0, 0, 1), new int3(0, 1, 1))
                };

                // Vertice index to relative position of cube's vertice.
                cubeCorners = new int3[8]
                {
                    new int3(0, 0, 0),
                    new int3(1, 0, 0),
                    new int3(1, 0, 1),
                    new int3(0, 0, 1),
                    new int3(0, 1, 0),
                    new int3(1, 1, 0),
                    new int3(1, 1, 1),
                    new int3(0, 1, 1)
                };

                // Courtesy Paul Bourke https://paulbourke.net/geometry/polygonise/
                triangulation = new int[4096]
                {
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1,
                    3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1,
                    3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1,
                    3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1,
                    9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1,
                    9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
                    2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1,
                    8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1,
                    9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1,
                    4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1,
                    3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1,
                    1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1,
                    4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1,
                    4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1,
                    9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1,
                    5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1,
                    2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1,
                    9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1,
                    0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1,
                    2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1,
                    10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1,
                    4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1,
                    5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1,
                    5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1,
                    9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1,
                    0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1,
                    1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1,
                    10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1,
                    8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1,
                    2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1,
                    7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1,
                    9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1,
                    2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1,
                    11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1,
                    9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1,
                    5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1,
                    11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1,
                    11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1,
                    1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1,
                    9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1,
                    5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1,
                    2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
                    0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1,
                    5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1,
                    6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1,
                    3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1,
                    6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1,
                    5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1,
                    1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
                    10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1,
                    6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1,
                    8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1,
                    7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1,
                    3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
                    5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1,
                    0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1,
                    9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1,
                    8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1,
                    5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1,
                    0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1,
                    6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1,
                    10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1,
                    10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1,
                    8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1,
                    1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1,
                    3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1,
                    0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1,
                    10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1,
                    3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1,
                    6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1,
                    9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1,
                    8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1,
                    3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1,
                    6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1,
                    0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1,
                    10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1,
                    10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1,
                    2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1,
                    7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1,
                    7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1,
                    2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1,
                    1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1,
                    11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1,
                    8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1,
                    0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1,
                    7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1,
                    10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
                    2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
                    6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1,
                    7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1,
                    2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1,
                    1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1,
                    10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1,
                    10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1,
                    0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1,
                    7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1,
                    6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1,
                    8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1,
                    9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1,
                    6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1,
                    4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1,
                    10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1,
                    8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1,
                    0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1,
                    1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1,
                    8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1,
                    10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1,
                    4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1,
                    10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1,
                    5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
                    11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1,
                    9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
                    6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1,
                    7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1,
                    3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1,
                    7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1,
                    9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1,
                    3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1,
                    6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1,
                    9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1,
                    1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1,
                    4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1,
                    7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1,
                    6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1,
                    3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1,
                    0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1,
                    6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1,
                    0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1,
                    11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1,
                    6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1,
                    5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1,
                    9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1,
                    1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1,
                    1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1,
                    10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1,
                    0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1,
                    5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1,
                    10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1,
                    11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1,
                    9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1,
                    7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1,
                    2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1,
                    8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1,
                    9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1,
                    9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1,
                    1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1,
                    9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1,
                    9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1,
                    5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1,
                    0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1,
                    10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1,
                    2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1,
                    0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1,
                    0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1,
                    9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1,
                    5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1,
                    3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1,
                    5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1,
                    8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1,
                    0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1,
                    9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1,
                    0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1,
                    1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1,
                    3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1,
                    4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1,
                    9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1,
                    11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1,
                    11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1,
                    2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1,
                    9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1,
                    3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1,
                    1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1,
                    4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1,
                    4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1,
                    0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1,
                    3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1,
                    3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1,
                    0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1,
                    9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1,
                    1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                };
            }
        }
    }
}