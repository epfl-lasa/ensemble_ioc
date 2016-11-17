# ensemble_ioc
A module implementing ensemble inverse optimal control and relevant examples

## Dependencies:

Numpy           (>= 1.11.1)

Scikit-learn    (>= 0.18.0)

matplotlib      (>= 1.5.1)

[gmr repository](https://github.com/navigator8972/gmr.git)


## An inverted pendulum example - target and learned cost-to-go
Run the test
```
import PyMDP_Pendulum as mdp_pendulum
import cPickle as cp

#load data
demo_trajs = cp.load(open('bin/training_data.pkl', 'rb'))
mdp_pendulum.PendulumMDPValueLearningTest(demo_trajs)
```
![](./fig/target_cost_to_go.png)
![](./fig/learned_cost_to_go.png)
