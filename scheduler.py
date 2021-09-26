

class Scheduler :
    def __init__(self, embedding_dim, warmup_steps, lr) :
        self.d_model = embedding_dim
        self.warmup_steps = warmup_steps
        self.lr = lr

    def __call__(self, epoch) :
        step_num = epoch+1
        arg1 = self.d_model**(-0.5)
        arg2 = min(step_num**(-0.5) , step_num*(self.warmup_steps**(-1.5)))
        val = (arg1*arg2) 
        return val / self.lr

