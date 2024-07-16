from src.cassie import CassieEnv


if __name__ == "__main__":
    env = CassieEnv(env_config=dict(render_mode="human"))
    env.reset()
    for i in range(1000):
        env.step(env.action_space.sample() / 2)
        if env.render() is None:
            break
    env.close()
