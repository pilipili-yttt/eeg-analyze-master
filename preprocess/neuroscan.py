import mne
import numpy as np
import matplotlib.pyplot as plt

name = 'ZCH'
vhdr_file = f"C:/Users/14152/Desktop/collaboration_measurement/data/{name}.vhdr"

# 读取脑电数据
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

# 从注释中提取事件
events, event_id = mne.events_from_annotations(raw)

# 指定的事件ID，根据实际情况调整
target_event_id = 10002

# 找到指定事件的索引
target_events = events[events[:, 2] == target_event_id]

# 分段保存和可视化
for i, event in enumerate(target_events):
    start = event[0]
    # 使用下一个事件的开始作为结束，如果是最后一个事件，则使用数据的最后一点
    if i+1 < len(target_events):
        end = target_events[i+1][0]
    else:
        end = raw.n_times
    
    # 分段
    segment, times = raw[:, start:end]
    
    # 保存为.npy文件，文件名需要根据实际情况调整
    np.save(f"{name}_{i}.npy", segment)
    
    # 可视化
    plt.figure(figsize=(10, 7))
    plt.plot(times, segment.T)
    plt.title(f"Segment {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
