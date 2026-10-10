#!/usr/bin/env python
"""
Script to measure FPS, FLOPs, and Params for a ELSA model.
Usage:
    python measure_model.py --config-file configs/uavid/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import time
import torch
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from elsa import add_maskformer2_config


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_fps(model, input_size=(1, 3, 512, 512), warmup=50, iterations=200):
    """Measure FPS with warmup"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)
    
    # Create input in detectron2 format
    inputs = [{"image": dummy_input[0]}]
    
    # Warmup
    print(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    print(f"Measuring FPS over {iterations} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(inputs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    fps = iterations / elapsed_time
    latency = elapsed_time / iterations * 1000  # ms
    
    return fps, latency


def measure_flops(model, input_size=(1, 3, 512, 512)):
    """Measure FLOPs using fvcore"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)
    
    # For detectron2 models, we need to use the backbone + sem_seg_head
    try:
        # Try to get FLOPs for backbone
        flops = FlopCountAnalysis(model.backbone, dummy_input)
        backbone_flops = flops.total()
        
        # Get feature maps for sem_seg_head
        with torch.no_grad():
            features = model.backbone(dummy_input)
        
        # This is approximate - full model FLOPs are harder to measure
        print(f"  Backbone FLOPs: {backbone_flops / 1e9:.2f} GFLOPs")
        
        return backbone_flops
    except Exception as e:
        print(f"  Warning: Could not compute exact FLOPs: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Measure model FPS/FLOPs/Params")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--weights", default=None, help="Path to model weights (optional)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[1024, 1024], help="Input size HxW")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Measurement iterations")
    args = parser.parse_args()
    
    # Setup
    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()
    
    # Load weights if specified
    if args.weights:
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(args.weights)
    
    device = next(model.parameters()).device
    print(f"\n{'='*60}")
    print(f"Model: {cfg.MODEL.META_ARCHITECTURE}")
    print(f"Backbone: {cfg.MODEL.BACKBONE.NAME}")
    print(f"Device: {device}")
    print(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
    print(f"{'='*60}\n")
    
    # 1. Count Parameters
    print("=" * 40)
    print("PARAMETERS")
    print("=" * 40)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total params:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 2. Measure FLOPs
    print("\n" + "=" * 40)
    print("FLOPs")
    print("=" * 40)
    h, w = args.input_size
    input_size = (1, 3, h, w)
    measure_flops(model, input_size)
    
    # 3. Measure FPS
    print("\n" + "=" * 40)
    print("SPEED (FPS)")
    print("=" * 40)
    fps, latency = measure_fps(model, input_size, args.warmup, args.iterations)
    print(f"  FPS:     {fps:.2f}")
    print(f"  Latency: {latency:.2f} ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Parameters: {total_params/1e6:.2f}M")
    print(f"  FPS:        {fps:.2f}")
    print(f"  Latency:    {latency:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
