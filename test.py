import mne

# 设置BrainVision文件的路径（这里我们只需要.vhdr文件的路径）
# 假设文件名为 'example.vhdr'
vhdr_file = "C:\\Users\\14152\\Desktop\\collaboration_measurement\\data\\LZ.vhdr"

raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
print(raw.info)

# 从数据中提取事件信息
events = mne.events_from_annotations(raw)

# events[0] 包含了事件数组，events[1] 包含了事件ID映射
events_array, event_id = events

# 打印事件ID映射
print(event_id)

# 查看前5个事件
print(events_array[:5])

# 使用 plot_events 方法可视化事件
mne.viz.plot_events(events_array, event_id=event_id, sfreq=raw.info['sfreq'])
