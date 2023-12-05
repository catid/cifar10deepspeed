# https://github.com/catid/UltraFastBERT/blob/main/training/cramming/architectures/fff.py

# This is the Fast Feedforward Network architecture from the UltraFastBERT paper

# Modified with a fan-out > 2 by using softmax for decisions

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class FFFFanout(nn.Module):
	def __init__(self, input_width, output_width, depth, parallel_size, fanout, activation=nn.GELU):
		super().__init__()

		self.input_width = input_width
		self.output_width = output_width
		self.depth = depth
		self.parallel_size = parallel_size
		self.fanout = fanout
		self.n_nodes = ((fanout ** (self.depth + 1) - 1) // 3) * fanout

		self.linear_in = nn.Linear(input_width, parallel_size * self.n_nodes, bias=True)
		self.linear_out = nn.Linear(parallel_size * self.n_nodes, output_width, bias=False)

		init_k = math.sqrt(1.0 / self.input_width)
		self.linear_in.weight.data = torch.empty((self.parallel_size * self.n_nodes, self.input_width)).uniform_(-init_k, +init_k)
		self.linear_in.bias.data = torch.empty((self.parallel_size * self.n_nodes)).uniform_(-init_k, +init_k)
		init_k2 = math.sqrt(1.0 / ((self.depth+1) * self.parallel_size))
		self.linear_out.weight.data = torch.empty((self.output_width, self.parallel_size * self.n_nodes)).uniform_(-init_k2, +init_k2)

		self.activation = activation()

	def forward(self, oldx: torch.Tensor) -> torch.Tensor:
		# x has shape (..., input_width)
		x = oldx.reshape(-1, self.input_width)
		# x has shape (batch_size, input_width)
		batch_size = x.shape[0]

		logits = self.linear_in(x) # (batch_size, parallel_size * n_nodes)
		activations = self.activation(logits) # (batch_size, parallel_size * n_nodes)

		# All activations, with final dimension for a single fanout-group
		activations = activations.view(batch_size, self.parallel_size, self.n_nodes // self.fanout, self.fanout)

		# Make a hard decision for each block of `fanout` nodes to decide where to move next
		decisions = torch.argmax(F.softmax(activations, dim=-1), dim=-1) # (batch_size, parallel_size, n_nodes/fanout)

		# Fill in the activation_mask with 1's where we want to select those
		with torch.no_grad():
			current_node = torch.zeros((batch_size, self.parallel_size), dtype=torch.long, device=x.device)
			activation_mask = torch.zeros_like(activations, dtype=x.dtype) # (batch_size, parallel_size, n_nodes/fanout, fanout)

			# We always activate the first layer:
			activation_mask[:, :, 0, :] = 1.0

			platform = 0
			for d in range(self.depth):
				# Gather how we will move next
				moves = torch.gather(decisions, 2, current_node.unsqueeze(-1)).squeeze(-1)
				next_platform = platform + self.fanout ** d
				current_node = next_platform + self.fanout * (current_node - platform) + moves

				# Scatter 1.0 values for the branch we selected
				index_tensor = current_node.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.fanout)
				activation_mask.scatter_(2, index_tensor, 1.0)

				platform = next_platform

		activations = activations * activation_mask # (batch_size, parallel_size, n_nodes/fanout, fanout)
		new_logits = self.linear_out(activations.flatten(1)) # (batch_size, output_width)

		ret = new_logits.reshape_as(oldx)

		return ret
	