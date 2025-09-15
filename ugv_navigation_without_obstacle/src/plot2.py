import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def plot_line_chart_from_file(file_path, num_lines=2000, title='Line Chart', xlabel='Index', ylabel='Value', 
                              line_width=2, use_moving_average=False, window_size=50, 
                              smooth=False, smooth_factor=0.5, save_path=None):
    """
    绘制折线图，可选移动平均和平滑处理
    smooth_factor: 平滑因子，越大曲线越平缓
    """
    # 读取数据
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            data.append(float(line.strip()))  # 假设数据为数值类型

    # 创建数据数组
    x = np.arange(1, len(data) + 1)
    y = np.array(data)

    # 可选：计算移动平均
    if use_moving_average:
        y = pd.Series(y).rolling(window=window_size).mean().bfill()

    # 可选：平滑处理
    if smooth and len(y) > 3:
        spline = UnivariateSpline(x, y)
        spline.set_smoothing_factor(smooth_factor)
        y = spline(x)

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=line_width)

    # 设置标题和轴标签
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # 纵轴自适应数据范围
    plt.ylim(min(y) * 0.95, max(y) * 1.05)

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存至 {save_path}")
    else:
        plt.show()


# 示例调用：
# plot_line_chart_from_file(
#     'data.txt', 
#     num_lines=2000, 
#     title='Smooth Line Chart', 
#     xlabel='Sample Index', 
#     ylabel='Sample Value', 
#     line_width=2, 
#     use_moving_average=False, 
#     smooth=True, 
#     smooth_factor=0.8, 
#     save_path='smooth_chart.png'
# )

# 保存Reward图像
plot_line_chart_from_file(
    'Your_Model_PathYour_Reward_File_Name', 
    num_lines=2000, 
    title='', 
    xlabel='Episode', 
    ylabel='Reward', 
    line_width=2.0, 
    use_moving_average=False, 
    smooth=True, 
    smooth_factor=31000, 
    save_path='reward.png'
    )
# 保存Success Rate图像
plot_line_chart_from_file(
    'Your_Model_PathYour_Success_Rate_File_Name', 
    num_lines=2000, 
    title='', 
    xlabel='Episode', 
    ylabel='Success Rate', 
    line_width=2.0, 
    use_moving_average=False, 
    smooth=True, 
    smooth_factor=25, 
    save_path='sr.png'
    )
