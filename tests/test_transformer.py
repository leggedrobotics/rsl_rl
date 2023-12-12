import unittest
import torch

from rsl_rl.modules import Transformer  # Assuming the Transformer class is in a module named my_module


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.input_size = 9
        self.output_size = 12
        self.hidden_size = 64
        self.block_count = 2
        self.context_length = 32
        self.head_count = 4
        self.batch_size = 10
        self.sequence_length = 16

        self.transformer = Transformer(
            self.input_size, self.output_size, self.hidden_size, self.block_count, self.context_length, self.head_count
        )

    def test_num_layers(self):
        self.assertEqual(self.transformer.num_layers, self.context_length // 2)

    def test_reset_hidden_state(self):
        hidden_state = self.transformer.reset_hidden_state(self.batch_size)
        self.assertIsInstance(hidden_state, tuple)
        self.assertEqual(len(hidden_state), 2)
        self.assertTrue(
            torch.equal(hidden_state[0], torch.zeros((self.transformer.num_layers, self.batch_size, self.hidden_size)))
        )
        self.assertTrue(
            torch.equal(hidden_state[1], torch.zeros((self.transformer.num_layers, self.batch_size, self.hidden_size)))
        )

    def test_step(self):
        x = torch.rand(self.sequence_length, self.batch_size, self.input_size)
        context = torch.rand(self.context_length, self.batch_size, self.hidden_size)

        out, new_context = self.transformer.step(x, context)

        self.assertEqual(out.shape, (self.sequence_length, self.batch_size, self.output_size))
        self.assertEqual(new_context.shape, (self.context_length, self.batch_size, self.hidden_size))

    def test_forward(self):
        x = torch.rand(self.sequence_length, self.batch_size, self.input_size)
        hidden_state = self.transformer.reset_hidden_state(self.batch_size)

        out, new_hidden_state = self.transformer.forward(x, hidden_state)

        self.assertEqual(out.shape, (self.sequence_length, self.batch_size, self.output_size))
        self.assertEqual(len(new_hidden_state), 2)
        self.assertEqual(new_hidden_state[0].shape, (self.transformer.num_layers, self.batch_size, self.hidden_size))
        self.assertEqual(new_hidden_state[1].shape, (self.transformer.num_layers, self.batch_size, self.hidden_size))


if __name__ == "__main__":
    unittest.main()
