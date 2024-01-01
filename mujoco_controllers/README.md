# Running Controller Examples

Simply run python script for <controller.py>, for all the apple fan boys/girls run command as follows: 

```python
mjpython <controller.py>
```

See the following [discussion](https://github.com/google-deepmind/mujoco/discussions/780) for why this is required. 

# Tuning Controllers for Task Domain
I currently tune these controller implementations for each particular task environment. The reason for this is that each environment has different physics settings (timesteps etc.) and task requirements. I hope to open source environment setups for tasks in future.
