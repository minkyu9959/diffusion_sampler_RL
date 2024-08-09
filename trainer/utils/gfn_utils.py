def get_exploration_std(
    epoch, exploratory, exploration_factor=0.1, exploration_wd=False
):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1.0 - epoch / 5000.0)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl
