from src.cassie import CassieEnv

import matplotlib.pyplot as plt
if __name__ == "__main__":
    env = CassieEnv(env_config=dict(render_mode="human"))
    plt.plot(env.von_mises_values_stance)
    plt.show()
    plt.savefig("von_mises_values_stance.png")
    env.reset()
    for i in range(1000):
        env.step(env.action_space.sample() / 5)
        if env.render() is None:
            break
    env.close()
