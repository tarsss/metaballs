using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Random = UnityEngine.Random;

namespace Tars.Metaballs
{
    [RequireComponent(typeof(MeshFilter))]
    public class Example : MonoBehaviour
    {
        [SerializeField] private float _isoLevel = 0.5f;
        [SerializeField] private bool _scheduleParallel;

        private Mesh _mesh;
        private MetaballsJobHandle _jobHandle;
        private NativeArray<Metaball> _metaballs;
        private float3[] _velocities;

        private void Awake()
        {
            _mesh = new Mesh();
            GetComponent<MeshFilter>().mesh = _mesh;

            _metaballs = new NativeArray<Metaball>(100, Allocator.Persistent);
            _velocities = new float3[100];

            for (int i = 0; i < _metaballs.Length; i++)
            {
                _metaballs[i] = new Metaball()
                {
                    position = Random.insideUnitSphere * 50f,
                    maxRadius = Random.Range(8, 20),
                    scale = Random.Range(0.75f, 1.5f)
                };
            }
        }

        private void Update()
        {
            if (_scheduleParallel)
            {
                _jobHandle.Complete();
                if (_jobHandle.IsScheduled)
                    _jobHandle.Apply();
            }

            var origin = new float3(Mathf.Sin(Time.time), Mathf.Cos(Time.time), Mathf.Cos(Time.time)) * 5f;
            for (int i = 0; i < _metaballs.Length; i++)
            {
                _velocities[i] += math.normalize(origin - _metaballs[i].position) * Time.deltaTime * 30f;
                var blob = _metaballs[i];
                blob.position += _velocities[i] * Time.deltaTime;
                _metaballs[i] = blob;
            }

            if (_scheduleParallel)
                _jobHandle = Metaballs.Schedule(_metaballs, _isoLevel, _mesh);
            else
                Metaballs.Run(_metaballs, _isoLevel, _mesh);
        }

        private void OnDisable()
        {
            if (_scheduleParallel)
                _jobHandle.Complete();
            _metaballs.Dispose();
        }
    }
}