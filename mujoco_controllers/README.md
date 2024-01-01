# Running Controller Examples

Simply run python script for <controller.py>, for all the apple fan boys/girls run command as follows: 

```python
mjpython <controller.py>
```

See the following [discussion](https://github.com/google-deepmind/mujoco/discussions/780) for why this is required. 

# Tuning Controllers for Task Domain
I currently tune these controller implementations for each particular task environment. The reason for this is that each environment has different physics settings (timesteps etc.) and task requirements. My current approach is to use genetic algorithms for this tuning process, it remains far from perfect in its current form. I hope to open source environments and tuning setups for tasks in future.
