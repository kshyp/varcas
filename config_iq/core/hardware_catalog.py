"""Hardware catalog management for cloud VM instances."""

import json
import os
from typing import List, Optional, Dict, Any
from .types import VMInstance, GPUSpec


class HardwareCatalog:
    """Manages hardware configurations from cloud providers."""
    
    def __init__(self, catalog_path: Optional[str] = None):
        """Initialize hardware catalog.
        
        Args:
            catalog_path: Path to catalog JSON file. If None, uses default.
        """
        if catalog_path is None:
            catalog_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'gcp_catalog.json'
            )
        
        self.catalog_path = catalog_path
        self.instances: List[VMInstance] = []
        self.metadata: Dict[str, Any] = {}
        self._load_catalog()
    
    def _load_catalog(self):
        """Load VM instances from catalog file."""
        with open(self.catalog_path, 'r') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        
        for inst_data in data.get('vm_instances', []):
            gpus = []
            for gpu_data in inst_data.get('gpus', []):
                gpus.append(GPUSpec(
                    type=gpu_data['type'],
                    count=gpu_data['count'],
                    vram_gb=gpu_data['vram_gb'],
                    compute_capability=gpu_data['compute_capability'],
                    tensor_cores=gpu_data['tensor_cores'],
                    fp16_tflops=gpu_data['fp16_tflops'],
                    bf16_tflops=gpu_data['bf16_tflops'],
                    fp8_tflops=gpu_data['fp8_tflops'],
                    int8_tflops=gpu_data['int8_tflops'],
                    memory_bw_gbps=gpu_data['memory_bw_gbps']
                ))
            
            self.instances.append(VMInstance(
                name=inst_data['name'],
                family=inst_data['family'],
                vcpus=inst_data['vcpus'],
                memory_gb=inst_data['memory_gb'],
                gpus=gpus,
                ondemand_price_usd=inst_data['ondemand_price_usd'],
                spot_price_usd=inst_data['spot_price_usd'],
                network_bw_gbps=inst_data['network_bw_gbps'],
                available_zones=inst_data.get('available_zones', [])
            ))
    
    def get_all_instances(self) -> List[VMInstance]:
        """Get all VM instances in the catalog."""
        return self.instances
    
    def get_instances_by_gpu_type(self, gpu_type: str) -> List[VMInstance]:
        """Get instances with specific GPU type."""
        return [i for i in self.instances if any(g.type == gpu_type for g in i.gpus)]
    
    def get_instances_by_gpu_count(self, min_gpus: int = 1, max_gpus: int = 8) -> List[VMInstance]:
        """Get instances with GPU count in range."""
        return [
            i for i in self.instances 
            if min_gpus <= sum(g.count for g in i.gpus) <= max_gpus
        ]
    
    def get_instances_by_vram(self, min_vram_gb: int) -> List[VMInstance]:
        """Get instances with at least specified total VRAM."""
        return [i for i in self.instances if i.total_vram_gb >= min_vram_gb]
    
    def get_instance_by_name(self, name: str) -> Optional[VMInstance]:
        """Get a specific instance by name."""
        for inst in self.instances:
            if inst.name == name:
                return inst
        return None
    
    def get_instances_by_zone(self, zone: str) -> List[VMInstance]:
        """Get instances available in a specific zone."""
        return [i for i in self.instances if i.is_available_in_zone(zone)]
    
    def get_instances_by_region(self, region: str) -> List[VMInstance]:
        """Get instances available in a specific region."""
        return [i for i in self.instances if i.is_available_in_region(region)]
    
    def get_available_zones(self) -> List[str]:
        """Get list of all available zones across all instances."""
        zones = set()
        for inst in self.instances:
            zones.update(inst.available_zones)
        return sorted(list(zones))
    
    def get_available_regions(self) -> List[str]:
        """Get list of all available regions."""
        zones = self.get_available_zones()
        regions = set()
        for z in zones:
            # Region is everything before the last hyphen
            parts = z.rsplit('-', 1)
            if len(parts) == 2:
                regions.add(parts[0])
        return sorted(list(regions))
    
    def filter_compatible_instances(
        self,
        model_vram_requirement_gb: float,
        min_gpu_count: int = 1,
        max_gpu_count: int = 8,
        quantization: str = "fp16",
        zone: Optional[str] = None,
        region: Optional[str] = None
    ) -> List[VMInstance]:
        """Filter instances that can fit the model with tensor parallelism.
        
        Args:
            model_vram_requirement_gb: Total VRAM required for the model
            min_gpu_count: Minimum number of GPUs
            max_gpu_count: Maximum number of GPUs
            quantization: Quantization type (affects effective VRAM)
        
        Returns:
            List of compatible VM instances
        """
        # Adjust VRAM requirement based on quantization
        vram_multipliers = {
            "fp32": 2.0,
            "fp16": 1.0,
            "bf16": 1.0,
            "int8": 0.5,
            "fp8": 0.5,
            "int4": 0.25
        }
        multiplier = vram_multipliers.get(quantization, 1.0)
        adjusted_model_size = model_vram_requirement_gb * multiplier
        
        compatible = []
        for inst in self.instances:
            gpu_count = sum(g.count for g in inst.gpus)
            if not (min_gpu_count <= gpu_count <= max_gpu_count):
                continue
            
            # Check zone/region availability
            if zone and not inst.is_available_in_zone(zone):
                continue
            if region and not inst.is_available_in_region(region):
                continue
            
            # Check if model fits with tensor parallelism
            # Try different TP sizes to find one that works
            gpu_vram = inst.max_vram_per_gpu_gb
            valid_tp_sizes = [tp for tp in [1, 2, 4, 8] if tp <= gpu_count]
            
            fits = False
            for tp_size in valid_tp_sizes:
                # Model is sharded across TP group
                model_per_gpu = adjusted_model_size / tp_size
                
                # Overhead: CUDA context + activations + safety margin
                cuda_overhead_gb = 1.5
                activation_buffer_gb = model_per_gpu * 0.10
                safety_margin_gb = 0.5
                total_per_gpu = model_per_gpu + cuda_overhead_gb + activation_buffer_gb + safety_margin_gb
                
                if gpu_vram >= total_per_gpu:
                    fits = True
                    break
            
            if fits:
                compatible.append(inst)
        
        return compatible
    
    def get_interconnect_info(self, gpu_type: str) -> Dict[str, Any]:
        """Get interconnect bandwidth and topology for a GPU type."""
        # Load from workload defaults
        defaults_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'workload_defaults.json'
        )
        with open(defaults_path, 'r') as f:
            data = json.load(f)
        
        return data.get('interconnect', {}).get(gpu_type, {})
    
    def get_tensor_parallel_efficiency(self, tp_size: int) -> float:
        """Get tensor parallel efficiency factor for a given TP size."""
        defaults_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'workload_defaults.json'
        )
        with open(defaults_path, 'r') as f:
            data = json.load(f)
        
        efficiency_map = data.get('vllm_overhead_factors', {}).get(
            'tensor_parallel_efficiency', {}
        )
        return efficiency_map.get(str(tp_size), 0.85)
    
    def get_tensor_parallel_overhead(self, gpu_type: str) -> float:
        """Get tensor parallel overhead for a GPU type."""
        interconnect = self.get_interconnect_info(gpu_type)
        return interconnect.get('tensor_parallel_overhead', 0.10)
    
    def calculate_effective_memory_bw(
        self,
        vm: VMInstance,
        tensor_parallel_size: int = 1
    ) -> float:
        """Calculate effective memory bandwidth considering tensor parallelism.
        
        In tensor parallelism, activations need to be communicated between GPUs,
        which reduces effective memory bandwidth.
        """
        total_bw = sum(g.memory_bw_gbps * g.count for g in vm.gpus)
        
        if tensor_parallel_size <= 1:
            return total_bw
        
        # Apply tensor parallel overhead
        gpu_type = vm.gpu_type
        overhead = self.get_tensor_parallel_overhead(gpu_type)
        
        # More GPUs = more communication overhead
        # NVLink has much less overhead than PCIe
        efficiency_factor = self.get_tensor_parallel_efficiency(tensor_parallel_size)
        
        return total_bw * efficiency_factor
    
    def calculate_effective_compute(
        self,
        vm: VMInstance,
        quantization: str = "fp16",
        tensor_parallel_size: int = 1
    ) -> float:
        """Calculate effective compute in TFLOPS.
        
        Args:
            vm: VM instance
            quantization: Quantization type
            tensor_parallel_size: Tensor parallel size
        """
        gpu = vm.gpus[0]
        
        # Get compute for quantization type
        compute_map = {
            "fp16": gpu.fp16_tflops,
            "bf16": gpu.bf16_tflops,
            "fp8": gpu.fp8_tflops,
            "int8": gpu.int8_tflops,
            "int4": gpu.int8_tflops * 2  # Approximate
        }
        compute_per_gpu = compute_map.get(quantization, gpu.fp16_tflops)
        
        total_compute = compute_per_gpu * sum(g.count for g in vm.gpus)
        
        # Apply tensor parallel efficiency
        if tensor_parallel_size > 1:
            efficiency = self.get_tensor_parallel_efficiency(tensor_parallel_size)
            total_compute *= efficiency
        
        return total_compute
