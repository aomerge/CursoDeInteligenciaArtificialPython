import unittest
from decay_schedule import LinearDecaySchedule

class TestLinearDecaySchedule(unittest.TestCase):
    def test_linear_decay_schedule(self):
        # Test case 1: initialValue > finalValue
        schedule = LinearDecaySchedule(1.0, 0.0, 10)
        self.assertEqual(schedule(0), 1.0)
        self.assertEqual(schedule(5), 0.5)
        self.assertEqual(schedule(10), 0.0)
        self.assertEqual(schedule(15), 0.0)  # step_num > max_steps, should still return finalValue

    def test_initial_value_greater_than_final_value(self):
        # Test 2: initialValue = finalValue        
        with self.assertRaises(AssertionError) as context:
            my_obj = LinearDecaySchedule(initialValue=100, finalValue=150, max_steps=10)
        self.assertTrue("initialValue should be greater than finalValue." in str(context.exception))

if __name__ == '__main__':
    unittest.main()