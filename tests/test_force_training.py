import unittest
import tempfile
from pathlib import Path
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Training extras required')
from berth_sim.training.force_model import ForceGenerator, ForceDiscriminator, fit_force_scaler
from berth_sim.training.model import energy_score
from berth_sim.training.train import load_model, predict


class ForceTrainingTests(unittest.TestCase):
    def test_constraints_gradients_and_checkpoint(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(1)
        rms=rng.uniform(100,10000,(20,3)).astype(np.float32)
        values=np.concatenate((rms,rms+rng.uniform(10,100,(20,3))),axis=-1).astype(np.float32)
        config={'task':'loads','hidden_size':8,'noise_size':3,'horizon_steps':5}
        scaler=fit_force_scaler(values)
        model=ForceGenerator(config,scaler)
        history=torch.from_numpy(values[:12].reshape(2,6,6))
        samples=model(history,4)
        self.assertEqual(samples.shape,(2,4,5,6))
        self.assertTrue((samples[...,3:]>=samples[...,:3]).all())
        self.assertTrue((samples>=0).all())
        target=history[:,:5]
        loss=energy_score(model.transform.score_space(samples),model.transform.score_space(target))
        critic=ForceDiscriminator(config,scaler)
        loss=loss+torch.nn.functional.softplus(-critic(history,samples[:,0])).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'best.pt'
            torch.save({'config':config,'scaler':scaler,'model_name':'gan_lstm','model_state':model.state_dict()},path)
            loaded,_=load_model(path)
            np.testing.assert_array_equal(predict(model,history.numpy(),4,seed=2),predict(loaded,history.numpy(),4,seed=2))
