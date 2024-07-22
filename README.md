# Investigating biologically plausible learning rules for solving credit assignment problem

Our project devoted to the credit assignment problem. We are particularly interested in biologically plausible learning rules because backpropagation seems not a very realistic rule for the brain due to the weight transfer problem. Ultimately we want to know/show if learning rules like weight perturbation, feedback alignment and predictive coding would also be appropriate rules for learning. And how these rules perform in more complicated scenarios. We want to investigate some properties of these algorithms and to do so we need to answer the following questions.
1. Do some algorithms perform better than others? And how do they perform compared with backprop?
2. What can we learn about the learning rules from the metrics measured during their training?
3. Are some algorithms better suited for complex tasks (online learning, non-stationary data)? If so, why?
4. Does the biological plausibility of learning rules correlate with their performance?
5. What are the potential benefits of combining weight perturbation with feedback alignment?
6. How might using predictive coding as a pre-training step, followed by another learning rule, improve neural network performance?
7. What advantages could hybrid approaches that incorporate both biologically plausible and non-biologically plausible components offer in machine learning?

In order to tackle these questions and reach the goal of the project we need to:
1. Implement algorithms of weight perturbation and feedback alignment.
2. Measure the performance of these algorithms using metrics such as convergence speed, final accuracy, sensitivity to learning rate and robustness against noise.
3. Measure other metrics (bias and variance of the gradients) in normal settings
4. Measure the performance in more complicated settings (batch size=1, non-stationary data)
5. Implement predictive coding and repeat the above steps
6. Combining weight perturbation with feedback alignment
7. Using predictive coding as a pre-training step followed by another learning rule
8. Exploring hybrid approaches that leverage both biologically plausible and non-biologically plausible components
