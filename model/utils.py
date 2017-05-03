import os

import torch
import torch.nn as nn


def save_checkpoint(m, o, steps, path):
	state = {
				'm_state_dict': m.state_dict(),
				'o_state_dict': o.state_dict(),
				'steps': steps
			}

	torch.save(state, path)


def load_checkpoint(path, m, o):
	print("[#] Loading model checkpoint from : {}".format(path))
	try:
		state = torch.load(path)
		loaded_m_state = state['m_state_dict']
		loaded_o_state = state['o_state_dict']
		steps = state['steps']

		try:
			m.load_state_dict(loaded_m_state)
		except:
			print("  [#] Partial model in ckpt. Loading ..")
			mstate = m.state_dict()
			keystobeupdated = set(mstate.keys()).intersection(set(loaded_m_state.keys()))
			for k in keystobeupdated:
				mstate[k] = loaded_m_state[k]
			m.load_state_dict(mstate)

		o.load_state_dict(loaded_o_state)
		print("[#] Loading successful : {}".format(steps))
		return steps
	except:
		print("[#] Loading Failed")