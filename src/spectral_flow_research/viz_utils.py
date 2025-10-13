
import matplotlib.pyplot as plt
def stamp(ax, text: str, loc: str = "lr", alpha: float = 0.6, fontsize: int = 8):
    if not text: return
    if loc == "lr": xy=(0.99,0.01); ha="right"; va="bottom"
    elif loc == "ll": xy=(0.01,0.01); ha="left"; va="bottom"
    elif loc == "ur": xy=(0.99,0.99); ha="right"; va="top"
    else: xy=(0.01,0.99); ha="left"; va="top"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va, alpha=alpha, fontsize=fontsize)
