#!/usr/bin/env python3
import torch
import argparse
import time
from timm.models import create_model
import DiffRate
import json

def speed_test(model, ntest=500, batchsize=16, x=None, use_fp16=False, **kwargs):
    """
    类似用户提供的speed_test函数
    
    Args:
        model: 要测试的模型
        ntest: 测试次数
        batchsize: 批次大小
        x: 输入张量，如果为None则自动生成
        use_fp16: 是否使用半精度
        **kwargs: 传递给model的额外参数
    
    Returns:
        throughput: 图像/秒的吞吐量
    """
    if x is None:
        # 尝试获取图像尺寸
        if hasattr(model, 'img_size'):
            img_size = model.img_size
        elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'img_size'):
            img_size = model.patch_embed.img_size
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
        else:
            img_size = (224, 224)  # 默认尺寸
        
        x = torch.rand(batchsize, 3, *img_size).cuda()
        if use_fp16:
            x = x.half()
    else:
        batchsize = x.shape[0]
    
    if use_fp16:
        model = model.half()
    
    model.eval()
    
    # 预热
    for _ in range(10):
        with torch.autocast('cuda', enabled=use_fp16):
            with torch.no_grad():
                model(x, **kwargs)

    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(ntest):
        with torch.autocast('cuda', enabled=use_fp16):
            with torch.no_grad():
                model(x, **kwargs)
                
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse
    return speed

def multi_speed_test(model, ntest=500, batchsize=16, repeat=3, use_fp16=False, verbose=True, **kwargs):
    """
    运行多次speed_test并返回统计结果
    
    Args:
        model: 要测试的模型
        ntest: 每轮测试次数
        batchsize: 批次大小
        repeat: 重复测试轮数
        use_fp16: 是否使用半精度
        verbose: 是否显示详细输出
        **kwargs: 传递给model的额外参数
    
    Returns:
        dict: 包含平均值、标准差、最小值、最大值的统计结果
    """
    import statistics
    
    if verbose:
        print(f"⏱️ Running {repeat} rounds of {ntest} iterations each...")
    
    throughputs = []
    for round_num in range(repeat):
        if verbose:
            print(f"🔄 Round {round_num + 1}/{repeat}...")
        
        throughput = speed_test(
            model=model,
            ntest=ntest,
            batchsize=batchsize,
            x=None,
            use_fp16=use_fp16,
            **kwargs
        )
        throughputs.append(throughput)
        
        if verbose:
            print(f"   📊 Round {round_num + 1} result: {throughput:.2f} images/second")
    
    # 计算统计信息
    avg_throughput = statistics.mean(throughputs)
    if len(throughputs) > 1:
        std_throughput = statistics.stdev(throughputs)
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
    else:
        std_throughput = 0
        min_throughput = max_throughput = avg_throughput
    
    results = {
        'average': avg_throughput,
        'std': std_throughput,
        'min': min_throughput,
        'max': max_throughput,
        'all_results': throughputs
    }
    
    if verbose:
        print(f"\n🎯 Statistical Results ({repeat} runs):")
        print(f"   📈 Average: {avg_throughput:.2f} ± {std_throughput:.2f} images/second")
        print(f"   📊 Range: {min_throughput:.2f} - {max_throughput:.2f} images/second")
    
    return results

def main():
    parser = argparse.ArgumentParser('DiffRate Throughput Benchmark')
    parser.add_argument('--model', default='deit_small_patch16_224', type=str,
                        help='Model name (deit_small_patch16_224, vit_large_patch16_mae, etc.)')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size for benchmarking')
    parser.add_argument('--runs', default=500, type=int,
                        help='Number of benchmark runs')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--use-fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--granularity', default=4, type=int,
                        help='Token granularity for DiffRate')
    parser.add_argument('--target_flops', type=float, default=2.9,
                        help='Target FLOPS for compression')
    parser.add_argument('--load_compression_rate', action='store_true',
                        help='Load compression rate from json file')
    parser.add_argument('--use_adjacent_merge', action='store_true',
                        help='Use adjacent merge algorithm (default: False)')
    parser.add_argument('--repeat', default=3, type=int,
                        help='Number of times to repeat the speed test (default: 3)')
    args = parser.parse_args()

    print(f"🚀 Benchmarking {args.model} with DiffRate")
    print(f"📊 Settings: batch_size={args.batch_size}, runs={args.runs}, repeat={args.repeat}, device={args.device}")
    print(f"🔍 Total iterations: {args.runs * args.repeat} ({args.runs} per round × {args.repeat} rounds)")
    
    device = torch.device(args.device)
    
    # 创建模型
    print("📝 Creating model...")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    
    # 应用DiffRate patch
    print(f"🔧 Applying DiffRate patch (adjacent_merge={args.use_adjacent_merge})...")
    if 'deit' in args.model:
        DiffRate.patch.deit(model, 
                           prune_granularity=args.granularity, 
                           merge_granularity=args.granularity, 
                           use_adjacent_merge=args.use_adjacent_merge)
    elif 'mae' in args.model:
        DiffRate.patch.mae(model, 
                          prune_granularity=args.granularity, 
                          merge_granularity=args.granularity, 
                          use_adjacent_merge=args.use_adjacent_merge)
    elif 'caformer' in args.model:
        DiffRate.patch.caformer(model, 
                               prune_granularity=args.granularity, 
                               merge_granularity=args.granularity)
    
    # 设置压缩率
    if args.load_compression_rate:
        print("📂 Loading compression rates...")
        model_name_dict = {
            'deit_tiny_patch16_224':'ViT-T-DeiT',
            'deit_small_patch16_224':'ViT-S-DeiT',
            'deit_base_patch16_224': 'ViT-B-DeiT',
            'vit_base_patch16_mae': 'ViT-B-MAE',
            'vit_large_patch16_mae': 'ViT-L-MAE',
            'vit_huge_patch14_mae': 'ViT-H-MAE',
            'caformer_s36':'CAFormer-S36',
        }
        
        try:
            with open('compression_rate.json', 'r') as f:
                compression_rate = json.load(f)
                model_name = model_name_dict[args.model]
                if str(args.target_flops) in compression_rate[model_name]:
                    prune_kept_num = eval(compression_rate[model_name][str(args.target_flops)]['prune_kept_num'])
                    merge_kept_num = eval(compression_rate[model_name][str(args.target_flops)]['merge_kept_num'])
                    model.set_kept_num(prune_kept_num, merge_kept_num)
                    print(f"✅ Set compression rate for {args.target_flops}G FLOPS")
                else:
                    print(f"⚠️  No compression rate found for {args.target_flops}G FLOPS, using default")
        except FileNotFoundError:
            print("⚠️  compression_rate.json not found, using default settings")
    
    # 确定输入尺寸
    if '224' in args.model:
        input_size = (3, 224, 224)
    elif '384' in args.model:
        input_size = (3, 384, 384)
    else:
        input_size = (3, 224, 224)  # 默认
    
    print(f"🔍 Input size: {input_size}")
    
    # 运行speed test（类似用户的speed_test函数）
    print("\n" + "="*50)
    print("🏃‍♂️ Running DiffRate Model Speed Test...")
    print("="*50)
    
    try:
        # 使用简化的speed_test函数，运行多次
        model.to(device)
        
        print(f"⏱️ Running {args.repeat} rounds of {args.runs} test iterations each...")
        
        throughputs = []
        for round_num in range(args.repeat):
            print(f"\n🔄 Round {round_num + 1}/{args.repeat}...")
            throughput = speed_test(
                model=model,
                ntest=args.runs,
                batchsize=args.batch_size,
                x=None,  # 自动生成输入
                use_fp16=args.use_fp16
            )
            throughputs.append(throughput)
            print(f"   📊 Round {round_num + 1} result: {throughput:.2f} images/second")
        
        print(f"\n✅ All speed tests completed!")
        
        # 计算统计信息
        import statistics
        avg_throughput = statistics.mean(throughputs)
        if len(throughputs) > 1:
            std_throughput = statistics.stdev(throughputs)
            min_throughput = min(throughputs)
            max_throughput = max(throughputs)
        else:
            std_throughput = 0
            min_throughput = max_throughput = avg_throughput
        
        print(f"\n🎯 Statistical Results ({args.repeat} runs):")
        print(f"   📈 Average Throughput: {avg_throughput:.2f} ± {std_throughput:.2f} images/second")
        print(f"   📊 Range: {min_throughput:.2f} - {max_throughput:.2f} images/second")
        print(f"   ⚡ Average Batch throughput: {avg_throughput / args.batch_size:.2f} batches/second")
        print(f"   🕐 Average time per image: {1000.0 / avg_throughput:.2f} ms")
        print(f"   📦 Average time per batch: {1000.0 * args.batch_size / avg_throughput:.2f} ms")
        
        if args.use_fp16:
            print(f"   🧠 Precision: FP16")
        else:
            print(f"   🧠 Precision: FP32")
        
        # 显示详细结果
        print(f"\n📋 Detailed Results:")
        for i, tp in enumerate(throughputs, 1):
            deviation = ((tp - avg_throughput) / avg_throughput) * 100
            print(f"   Round {i}: {tp:.2f} img/s ({deviation:+.1f}%)")
            
        return avg_throughput
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None

if __name__ == '__main__':
    main()

# 简单的使用示例：
"""
# 示例用法1：单次测试
import torch
from timm.models import create_model
import DiffRate
from benchmark_example import speed_test, multi_speed_test

# 创建并patch模型
model = create_model('deit_small_patch16_224', pretrained=True)
DiffRate.patch.deit(model, prune_granularity=4, merge_granularity=4, use_adjacent_merge=False)

# 设置压缩率（可选）
# model.set_kept_num(prune_kept_num, merge_kept_num)

# 单次快速测试
speed = speed_test(model, ntest=100, batchsize=16)
print(f"Single test: {speed:.2f} images/second")

# 多次测试获得更稳定结果
results = multi_speed_test(model, ntest=100, batchsize=16, repeat=3)
print(f"Average: {results['average']:.2f} ± {results['std']:.2f} images/second")
print(f"Range: {results['min']:.2f} - {results['max']:.2f}")
""" 