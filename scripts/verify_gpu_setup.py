#!/usr/bin/env python3
"""
Verify GPU setup for RTX 4090s.
"""
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    logger.info("Checking PyTorch CUDA support...")
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Check your drivers and PyTorch installation.")
        return False
    
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA devices.")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"Device {i}: {props.name} | VRAM: {props.total_memory / 1024**3:.2f} GB")
        
        # Check for RTX 4090 specific architecture (Compute Capability 8.9)
        major, minor = torch.cuda.get_device_capability(i)
        logger.info(f"  Compute Capability: {major}.{minor}")
        if major < 8:
            logger.warning(f"  Warning: Device {i} might be older architecture.")
            
    # Check FAISS GPU support
    logger.info("\nChecking FAISS GPU support...")
    try:
        import faiss
        res = faiss.StandardGpuResources()
        logger.info("FAISS GPU resources initialized successfully.")
        
        # Create a dummy index on GPU
        d = 64
        index = faiss.IndexFlatL2(d)
        if device_count > 1:
            logger.info(f"Testing index_cpu_to_all_gpus with {device_count} GPUs...")
            gpu_index = faiss.index_cpu_to_all_gpus(index)
        else:
            logger.info("Testing index_cpu_to_gpu...")
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            
        logger.info("FAISS GPU index created successfully.")
        
    except ImportError:
        logger.error("faiss-gpu not installed!")
        return False
    except Exception as e:
        logger.error(f"FAISS GPU test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if check_gpu():
        logger.info("\n✅ System is ready for RTX 4090 experiments!")
        sys.exit(0)
    else:
        logger.error("\n❌ System check failed.")
        sys.exit(1)
