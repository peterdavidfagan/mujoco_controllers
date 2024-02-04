# Running Controller Examples

Simply run python scripts after activating poetry environment, 

```
python <controller.py>
```

if you are on MacOS you will need to run the command as follows in order to load the interactive viewer: 

```python
mjpython <controller.py>
```

See the following [discussion](https://github.com/google-deepmind/mujoco/discussions/780) for further details on interactive viewer on MacOS. 

# Tuning Controllers for Task Domain
I currently tune these controller implementations for each particular task environment. The reason for this is that each environment has different physics settings (timesteps etc.) and task requirements. My current approach is to use genetic algorithms for this tuning process, it remains far from perfect in its current form. I hope to open source environments and tuning setups for tasks in future.

# Operational Space Control (Current p2p qualitative performance)
https://github.com/peterdavidfagan/mujoco_controllers/assets/42982057/7e2ac905-d217-4851-8bc8-7e31a5336141

# Differential Inverse Kinematics (Current p2p qualitative performance)
https://github.com/peterdavidfagan/mujoco_controllers/assets/42982057/5decb8a9-6e73-41e9-a289-9b31037a3acc

