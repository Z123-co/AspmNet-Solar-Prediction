"""
================================================================================
AspmNet: Quick Start Script
直接运行此脚本，自动使用指定的数据文件
================================================================================
"""

import sys
import os

# 设置你的数据文件路径
data_path = r"C:/Users/86131/OneDrive/桌面/season/Alice_Springs_2018.csv"

# 验证文件是否存在
if not os.path.exists(data_path):
    print("❌ 错误: 找不到数据文件!")
    print(f"   路径: {data_path}")
    sys.exit(1)

print("✓ 数据文件找到!")
print(f"✓ 路径: {data_path}")

# 执行主程序
if __name__ == "__main__":
    # 导入主程序
    from aspmnet_main import (
        Config, SolarDataLoader, TimeSeriesDecomposition,
        AspmNet, AspmNetTrainer, plot_results
    )
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import argparse
    from datetime import datetime
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "AspmNet: Solar PV Power Prediction - Quick Start".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # 配置参数
    EPOCHS = 50
    BATCH_SIZE = 32
    SEQ_LENGTH = 96
    LEARNING_RATE = 1e-3
    
    print(f"\n✓ 启动参数:")
    print(f"  • 数据文件: {data_path}")
    print(f"  • Epochs: {EPOCHS}")
    print(f"  • Batch size: {BATCH_SIZE}")
    print(f"  • Sequence length: {SEQ_LENGTH}")
    print(f"  • Learning rate: {LEARNING_RATE}")
    print(f"  • Device: {Config.DEVICE}")
    
    # ===== Step 1: Load data =====
    print("\n" + "="*80)
    print("开始处理数据...")
    print("="*80)
    
    loader = SolarDataLoader(data_path, test_size=Config.TEST_SIZE, val_size=Config.VAL_SIZE)
    data = loader.get_processed_data(seq_length=SEQ_LENGTH)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['scaler']
    
    print(f"\n✓ 数据加载完成!")
    print(f"  • 训练集: {X_train.shape[0]} 样本")
    print(f"  • 验证集: {X_val.shape[0]} 样本")
    print(f"  • 测试集: {X_test.shape[0]} 样本")
    
    # ===== Step 2: Decompose =====
    print("\n" + "="*80)
    print("进行时间序列分解...")
    print("="*80)
    
    decomposer = TimeSeriesDecomposition(window_size=Config.DECOMPOSE_WINDOW)
    X_train_flat = X_train.flatten()
    decomp_result = decomposer.decompose(X_train_flat.reshape(-1, 1))
    
    # ===== Step 3: Create data loaders =====
    print("\n" + "="*80)
    print("创建数据加载器...")
    print("="*80)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✓ 数据加载器创建完成!")
    print(f"  • 训练批次: {len(train_loader)}")
    print(f"  • 验证批次: {len(val_loader)}")
    print(f"  • 测试批次: {len(test_loader)}")
    
    # ===== Step 4: Initialize model =====
    print("\n" + "="*80)
    print("初始化 AspmNet 模型...")
    print("="*80)
    
    model = AspmNet(input_size=1, seasonal_hidden=Config.SEASONAL_HIDDEN, 
                   trend_hidden=Config.TREND_HIDDEN)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型初始化完成!")
    print(f"  • 总参数数: {total_params:,}")
    print(f"  • 可训练参数数: {trainable_params:,}")
    
    # ===== Step 5: Train =====
    print("\n" + "="*80)
    print("开始训练模型...")
    print("="*80)
    
    trainer = AspmNetTrainer(model, device=Config.DEVICE, learning_rate=LEARNING_RATE)
    trainer.train(train_loader, val_loader, epochs=EPOCHS, patience=Config.PATIENCE)
    
    # ===== Step 6: Test =====
    predictions, targets, metrics = trainer.test(test_loader)
    
    # ===== Step 7: Visualize =====
    print("\n" + "="*80)
    print("生成结果可视化...")
    print("="*80)
    
    plot_results(predictions, targets, scaler)
    
    # ===== Step 8: Save metrics =====
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    with open(Config.METRICS_PATH, 'w', encoding='utf-8') as f:
        f.write("AspmNet Solar Power Prediction - Test Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Sequence Length: {SEQ_LENGTH}\n")
        f.write(f"Epochs Trained: {len(trainer.train_losses)}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  MSE:  {metrics['mse']:.6f}\n")
        f.write(f"  MAE:  {metrics['mae']:.6f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
    
    # ===== Summary =====
    print("\n" + "="*80)
    print("✓ 训练完成!")
    print("="*80)
    print(f"\n📊 最终结果:")
    print(f"  • 训练轮数: {len(trainer.train_losses)}")
    print(f"  • 最佳验证损失: {min(trainer.val_losses):.6f}")
    print(f"  • 测试 RMSE: {metrics['rmse']:.6f} kW")
    print(f"  • 测试 MAPE: {metrics['mape']:.2f}%")
    
    print(f"\n📁 输出文件:")
    print(f"  • 模型: {Config.MODEL_PATH}")
    print(f"  • 指标: {Config.METRICS_PATH}")
    print(f"  • 图表: {Config.PLOT_PATH}")
    
    print("\n" + "="*80 + "\n")
