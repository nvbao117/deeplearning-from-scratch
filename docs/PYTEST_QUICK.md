# Pytest — 30 Giây Học Xong 🚀

## Nhớ 3 Bước Thôi

### 1️⃣ **Viết test**
```python
def test_something():
    assert 1 + 1 == 2
```

### 2️⃣ **Chạy**
```bash
pytest                    # chạy tất cả
pytest -v                 # chi tiết
pytest -k sigmoid         # chỉ sigmoid
```

### 3️⃣ **Fixture (nếu cần setup)**
```python
@pytest.fixture
def data():
    return np.array([1, 2, 3])

def test_with_data(data):
    assert len(data) == 3
```

---

## 5 Assert Dùng 90% Thời Gian

```python
assert x == 5                          # bằng
assert np.isclose(x, 3.14, atol=0.01)  # gần bằng
np.testing.assert_array_equal(a, b)    # mảng bằng
np.testing.assert_allclose(a, b)       # mảng gần bằng

with pytest.raises(ValueError):        # test exception
    bad_func()
```

---

## Parametrize — Chạy Test Nhiều Lần (Dùng Rất Nhiều)

```python
@pytest.mark.parametrize("x,y", [(1,2), (3,4), (5,6)])
def test_add(x, y):
    assert x + y > 0
```
→ Chạy test 3 lần với 3 input khác nhau

---

## Lệnh Chạy Thường Dùng Nhất

```bash
pytest                          # chạy hết
pytest -v                       # verbose
pytest test_file.py::test_func  # test cụ thể
pytest -x                       # stop ở fail đầu
pytest -k "sigmoid"             # chỉ "sigmoid"
pytest -s                       # show print()
```

---

## Conftest.py (Shared Fixture)

```python
# test/conftest.py — tất cả test trong folder test/ dùng được
import pytest
import numpy as np

@pytest.fixture
def zeros():
    return np.zeros(10)

# Dùng: def test_foo(zeros):
```

---

## Dễ Nhầm Nhất

| ❌ Sai | ✅ Đúng |
|--------|--------|
| `def test():` | `def test_foo():` |
| `self.assertEqual()` | `assert x == y` |
| Quên truyền fixture | `def test_foo(my_fixture):` |
| Float: `0.1+0.2==0.3` | `np.isclose(0.1+0.2, 0.3)` |

---

## Copy-Paste Template

```python
import pytest
import numpy as np
from activation_functions import sigmoid

# Test đơn giản
def test_sigmoid_zero():
    assert sigmoid(0.0) == 0.5

# Test nhiều input
@pytest.mark.parametrize("x,expected", [(0, 0.5), (-5, 0)])
def test_sigmoid_range(x, expected):
    assert np.isclose(sigmoid(x), expected, atol=0.01)

# Test với setup
@pytest.fixture
def x_array():
    return np.linspace(-5, 5, 100)

def test_with_array(x_array):
    y = sigmoid(x_array)
    assert y.shape == x_array.shape
```

---

## Chỉ Cần Nhớ:

1. **Hàm test** bắt đầu `test_`
2. **Assert** = condition kiểm tra
3. **Parametrize** = chạy nhiều lần
4. **Fixture** = setup data
5. `pytest -v` = chạy + xem chi tiết

**Xong! 🎉**
