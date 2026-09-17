import unittest
from app import app, model, encoder

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = app.app_context()
        self.ctx.push()
        self.client = app.test_client()

    def tearDown(self):
        self.ctx.pop()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CASEFILE', response.data)

    def test_prediction_success(self):
        response = self.client.post('/predict', data={
            'user_id': 'USER_001',
            'date': '2026-09-15',
            'time': '14:30'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'19.', response.data)
        self.assertIn(b'72.', response.data)

    def test_prediction_invalid_user(self):
        response = self.client.post('/predict', data={
            'user_id': 'USER_999',
            'date': '2026-09-15',
            'time': '14:30'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'not found in dataset', response.data)

if __name__ == '__main__':
    unittest.main()
