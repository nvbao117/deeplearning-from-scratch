# Pytest Cheat Sheet — Cách Dễ Nhớ

## 🚀 Cơ Bản (Nhớ 3 thứ này đủ rồi)

```python
import pytest

# 1️⃣ VIẾT TEST: Hàm bắt đầu với test_
def test_sigmoid_at_zero():
    assert sigmoid(0.0) == 0.5  # Đơn giản — chỉ assert thôi!

# 2️⃣ CHẠY TEST
# pytest test_file.py          # chạy cái file
# pytest -v                    # chi tiết từng test
# pytest -k sigmoid            # chỉ chạy test có "sigmoid"

# 3️⃣ FIXTURE: Setup data một lần dùng nhiều lần
@pytest.fixture
def my_data():
    return np.array([1, 2, 3])

def test_with_data(my_data):
    assert len(my_data) == 3
```

---

## 📋 Assert — Điều Kiện Test

```python
# So sánh
assert x == 5                          # bằng
assert x != 5                          # không bằng
assert x > 5                           # lớn hơn

# Mảng NumPy
import numpy as np
np.testing.assert_array_equal(a, b)    # mảng bằng nhau
np.testing.assert_allclose(a, b)       # gần bằng (cho float)

# String
assert "hello" in "hello world"
assert name.startswith("test")

# Exceptions
def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0

# Gần bằng (dùng cho float)
import math
assert math.isclose(3.14159, 3.14, abs_tol=0.01)
```

---

## 🎯 Parametrize — Chạy Test Nhiều Lần (Tiết Kiệm Code)

```python
# Thay vì viết 3 test hàm → viết 1 hàm, chạy 3 lần

@pytest.mark.parametrize("x,expected", [
    (0.0, 0.5),
    (-5.0, 0.0),
    (5.0, 1.0),
])
def test_sigmoid(x, expected):
    assert sigmoid(x) ≈ expected

# Chạy:
# pytest test_file.py::test_sigmoid -v
```

---

## 🔧 Fixture — Share Setup Giữa Tests

```python
@pytest.fixture
def sample_input():
    """Setup trước test"""
    x = np.array([-1, 0, 1])
    yield x  # Test dùng x
    # Cleanup sau test (nếu cần)

def test_with_fixture(sample_input):
    result = relu(sample_input)
    assert result.shape == sample_input.shape

# Fixture scope:
@pytest.fixture(scope="session")  # Dùng 1 lần cho tất cả test
def expensive_setup():
    return load_big_model()

@pytest.fixture(scope="function")  # (default) Dùng lại mỗi test
def fresh_data():
    return np.random.randn(100)
```

---

## 📂 Conftest.py — Fixture Dùng Chung Cho Cả Folder

```python
# test/conftest.py
import pytest
import numpy as np

@pytest.fixture
def zeros():
    return np.zeros((10, 5))

@pytest.fixture
def ones():
    return np.ones((10, 5))

# Bây giờ tất cả test trong folder test/ dùng được:
# test/test_1.py, test/test_2.py, v.v.
```

---

## 🏃 Lệnh Chạy Nhanh

```bash
# Chạy tất cả
pytest

# Verbose (chi tiết)
pytest -v

# Chỉ test có "sigmoid"
pytest -k sigmoid

# Chỉ test file nào đó
pytest test_activations.py

# Stop ở test đầu fail (thay vì chạy hết)
pytest -x

# Chạy 4 test trước đó fail
pytest --lf

# Hiển thị print() trong test
pytest -s

# Short traceback (không hiển thị dài dòng)
pytest --tb=short

# Test cụ thể
pytest test_file.py::TestClass::test_method
pytest test_file.py::test_function
```

---

## 🎭 Mark — Đặt Tag Cho Test

```python
# Đánh dấu test slow
@pytest.mark.slow
def test_expensive():
    time.sleep(5)

# Chạy chỉ test slow: pytest -m slow
# Chạy ngoài test slow: pytest -m "not slow"

# Đánh dấu test skip (bỏ qua)
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

# Đánh dấu test xfail (expected fail)
@pytest.mark.xfail
def test_buggy_function():
    assert broken_func() == 5  # Fail là ok
```

---

## 📊 Xem Coverage (% Code Chạy Qua Test)

```bash
pip install pytest-cov

# Xem % code chạy qua test
pytest --cov=activation_functions test/

# HTML report (mở trong browser)
pytest --cov=activation_functions --cov-report=html test/
# Mở: htmlcov/index.html
```

---

## 🚨 Sai Lầm Phổ Biến & Cách Sửa

| Sai | Đúng |
|-----|------|
| `def test():` (không có test_) | `def test_something():` |
| `unittest.assertEqual()` | `assert x == y` |
| `@pytest.fixture` không truyền vào hàm | `def test_foo(my_fixture):` |
| Quên `import pytest` | `import pytest` |
| `assert 0.1 + 0.2 == 0.3` (float sai) | `np.testing.assert_allclose([0.1+0.2], [0.3])` |

---

## 📝 Template Test Đơn Giản

```python
import pytest
import numpy as np
from activation_functions import sigmoid, d_sigmoid

def test_sigmoid_zero():
    """Test sigmoid tại x=0"""
    assert sigmoid(0.0) == 0.5

@pytest.mark.parametrize("x,expected", [(-10, 0), (0, 0.5), (10, 1)])
def test_sigmoid_range(x, expected):
    """Test sigmoid ở nhiều điểm"""
    assert np.isclose(sigmoid(x), expected, atol=0.01)

@pytest.fixture
def x_values():
    return np.linspace(-5, 5, 100)

def test_derivative(x_values):
    """Test đạo hàm"""
    numerical = (sigmoid(x_values + 1e-5) - sigmoid(x_values - 1e-5)) / 2e-5
    analytical = d_sigmoid(x_values)
    np.testing.assert_allclose(numerical, analytical, atol=1e-4)
```

---

## 🎓 Tóm Tắt — Câu Hỏi/Trả Lời

**Q: Test file đặt ở đâu?**  
A: Folder `test/` hoặc `tests/`, tên file bắt đầu `test_`

**Q: Fixture scope="session" có nghĩa gì?**  
A: Setup 1 lần cho tất cả test (không reset giữa các test)

**Q: Chạy test cụ thể như nào?**  
A: `pytest test_file.py::test_function_name -v`

**Q: Coverage là gì?**  
A: % dòng code chạy qua test (cao = code test kỹ)

**Q: Khi nào dùng unittest vs pytest?**  
A: Pytest dễ hơn, dùng pytest (trừ khi code cũ dùng unittest)
