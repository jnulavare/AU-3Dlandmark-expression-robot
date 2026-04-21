# AU-3Dmarkland-expression-robot
baseline:382->24->30
Compare1:382->30
Compare2:shared head->grouped head
Compare3:382->learned-latent24->30
Compare4：learnedlatent24->base+brow mouth residual head->30
Compare5(B1)：learnedlatent24->base+brow jaw+mouth residual head->30
Compare6(B1vnext):
Pre-LN Residual MLP Block
residual scale方案，原来是： y=ybase+Δybrow+Δymouth，
现在是y=ybase+αbΔybrow+αmΔymouth，
alpha_b, alpha_m 是可学习标量 初值设为 1.0。让模型自己学修正强度
使用SmoothL1  EMA使用。
