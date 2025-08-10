import matplotlib.pyplot as plt

iterations = ['random', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
calvin_net = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
hobbes_net = [-1199, -1199, -816, -723, -590, -287, -185, -183, -192, -163, -140, -141, -93, -85, -83, -72, -52, -45, -42, -42]

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(range(len(iterations)), calvin_net, 'o-', color='red', linewidth=2.5, markersize=7, label='Calvin Net')
ax.plot(range(len(iterations)), hobbes_net, 'o-', color='blue', linewidth=2.5, markersize=7, label='Hobbes Net')

ax.set_title('Hobbes Net Progression', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Iterations', fontsize=14)
ax.set_ylabel('Elo', fontsize=14)

ax.set_xticks(range(len(iterations)))
ax.set_xticklabels(iterations, fontsize=11)

ax.set_ylim(-1400, 200)
y_ticks = [0, -200, -400, -600, -800, -1000, -1200]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks, fontsize=11)

ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

ax.legend(fontsize=12, loc='lower right')

fig.text(0.5, 0.06, 'https://github.com/kelseyde/hobbes-chess-engine/blob/feature/selfgen/network_history.txt',
         fontsize=9, ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
