import jax
import jax.numpy as jnp


class RectFlow():
    
    def __init__(
        self, num_timesteps: int = 1000, num_samplesteps: int = 50, 
        noise_schedule: str = "linear", 
        **kwargs # TODO(guandao): bad code, needed bc we mix loss cfgs :(
    ):
        self.num_timesteps = num_timesteps
        self.num_samplesteps = num_samplesteps
        self.noise_schedule = noise_schedule
        
    def sample_timesteps(self, rng, x): 
        """
        Sample timesteps for diffusion process.
        Args:
            rng (jax.random.PRNGKey): Random number generator key.
            x (jax.numpy.ndarray): Input array.
        Returns:
            jax.numpy.ndarray: Array of sampled timesteps. 
                               Min 0, Max num_timesteps.
        Raises:
            NotImplementedError: If noise_schedule is not defined.
        """
        match self.noise_schedule:
            case "linear":
                t = jax.random.randint(
                    rng, shape=(x.shape[0],), 
                    minval=0, maxval=self.num_timesteps)
            case "lognorm": # SD3
                t = jax.nn.sigmoid(jax.random.normal(rng, shape=(x.shape[0],)))
                t = t * self.num_timesteps
            case _:
                raise NotImplemented
        return t
                
    def training_losses(
        self, model, rng, x_0, model_kwargs=None, noise=None, t=None):
        b = x_0.shape[0]
        if model_kwargs is None:
            model_kwargs = {}
        x_1 = noise
        if x_1 is None:
            rng, spl = jax.random.split(rng)
            x_1 = jax.random.normal(spl, shape=x_0.shape)
        
        if t is None:
            rng, spl = jax.random.split(rng)
            t = self.sample_timesteps(spl, x_0)
        
        # TODO: Check the noise schedule!   
        t_scale = t.reshape(b, *([1] * len(x_0.shape[1:]))) / float(self.num_timesteps) 
        x_t = x_0 + t_scale * (x_1 - x_0)
        
        # v_t = model(psi_t, t, conditioning)
        model_aux = {}
        model_output = model(x_t, t, **model_kwargs)
        if "return_aux" in model_kwargs and model_kwargs["return_aux"]:
            model_output, model_aux = model_output
        loss = ((model_output - x_1 + x_0) ** 2).mean()
        return {"mse": loss, "loss": loss}, t, model_aux
            
    def p_sample_loop(
        self,
        rng,
        model_fn,
        shape,
        noise=None,
        clip_denoised=False,
        model_kwargs=None,
        progress=False
    ):
        if model_kwargs is None:
            model_kwargs = {}
    
        if noise is not None:
            img = noise
        else:
            rng, spl = jax.random.split(rng)
            img = jax.random.normal(spl, shape=shape)
    
        B = img.shape[0]
        dt = 1.0 / self.num_samplesteps 
        t_lst = jnp.linspace(1, dt, self.num_samplesteps)
        
        def body(x_t, t):
            # cast back to [0, #num_timesteps]
            t = jnp.ones((B,)) * t * self.num_timesteps 
            assert t.shape == (B,)
            model_output = model_fn(x_t, t, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output, _ = model_output
            x_t_af = x_t - dt * model_output
            return x_t_af, {"v_t": model_output, "x_t": x_t, "x_t+1": x_t_af}
        
        out, _ = jax.lax.scan(body, img, t_lst)
        return out