import matplotlib.pyplot as plt
import numpy as np

labels = ['课程目标1', '课程目标2', '课程目标3']
values = [85, 72, 90]
colors = ['#4A90E2', '#5FA8D3', '#BFE4FF']

fig, ax = plt.subplots(figsize=(4.8, 3.6))
bars = ax.bar(labels, values, color=colors, width=0.48,
              edgecolor='white', linewidth=1.2, zorder=3)

# 圆角柱
for b in bars:
    b.set_linestyle('-')
    b.set_joinstyle('round')

# 数据标签
for v in values:
    ax.text(labels[values.index(v)], v+1.5, f'{v}%', ha='center', va='bottom',
            fontsize=11, fontweight='semibold')

# 坐标轴
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([f'{i}%' for i in range(0, 101, 20)])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.grid(axis='y', linestyle='-', linewidth=0.5, color='#F0F0F0', zorder=0)

# 标题
ax.set_title('课程目标达成度', fontsize=14, pad=16, fontweight='bold')

plt.tight_layout()
plt.savefig('course_objectives_bar.png', dpi=300)
plt.show()