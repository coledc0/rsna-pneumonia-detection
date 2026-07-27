import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import io
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_model():
    '''A fake YOLO model that returns a predictable, canned result.'''
    model = MagicMock()

    fake_box = MagicMock()
    fake_box.xyxy = [MagicMock(tolist=lambda: [10, 20, 100, 200])]
    fake_box.conf = [0.85]

    fake_result = MagicMock()
    fake_result.boxes = [fake_box]

    model.return_value = [fake_result]
    return model


@pytest.fixture
def test_image_bytes():
    '''A tiny valid PNG image in memory, used to simulate an upload.'''
    img = Image.new('L', (1024, 1024), color=128)  # grayscale, gray fill
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


@pytest.fixture
def client(mock_model, tmp_path):
    '''
    A TestClient with the model pre-injected, bypassing the real
    lifespan startup (no S3 download, no GPU, no real weights needed).
    '''
    fake_weights_path = tmp_path / 'best.pt'
    fake_weights_path.touch()  # create an empty file so any existence checks pass

    with patch('api.main.load_model', return_value=mock_model), \
         patch('api.main.find_best_run', return_value=fake_weights_path):

        from api.main import app, model_state
        model_state['model'] = mock_model
        model_state['model_path'] = 'mocked/path/best.pt'

        with TestClient(app) as test_client:
            yield test_client

        model_state.clear()


class TestHealthEndpoints:

    def test_root_returns_ok(self, client):
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert data['model_loaded'] is True

    def test_health_returns_ok(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'


class TestPredictEndpoint:

    def test_predict_with_valid_png(self, client, test_image_bytes):
        response = client.post(
            '/predict',
            files={'file': ('test.png', test_image_bytes, 'image/png')}
        )
        assert response.status_code == 200

        data = response.json()
        assert 'predicted' in data
        assert 'confidence' in data
        assert 'boxes' in data
        assert data['filename'] == 'test.png'

    def test_predict_rejects_invalid_file_type(self, client):
        fake_text_file = b'this is not an image'
        response = client.post(
            '/predict',
            files={'file': ('test.txt', fake_text_file, 'text/plain')}
        )
        assert response.status_code == 400

    def test_predict_response_has_correct_box_structure(self, client, test_image_bytes):
        response = client.post(
            '/predict',
            files={'file': ('test.png', test_image_bytes, 'image/png')}
        )
        data = response.json()

        if data['n_detections'] > 0:
            box = data['boxes'][0]
            assert 'x1' in box
            assert 'y1' in box
            assert 'x2' in box
            assert 'y2' in box
            assert 'confidence' in box