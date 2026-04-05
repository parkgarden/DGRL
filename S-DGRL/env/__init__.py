from gym.envs.registration import register

# Registrar for the gym environment
# https://www.gymlibrary.ml/content/environment_creation/ for reference
register(
    id='fjsp-v0',  # Environment name (including version number)
    entry_point='env.fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)

register(
    id='fjsp-v3',  # Environment name (including version number)
    entry_point='env.fjsp_env_dynv3:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)

register(
    id='fjspBA-v3',  # Environment name (including version number)
    entry_point='env.fjsp_env_dynv3BA:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)

register(
    id='fjspBA-v0',  # Environment name (including version number)
    entry_point='env.fjsp_envBA:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)