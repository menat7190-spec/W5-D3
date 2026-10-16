
import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

n_samples = 500

feature_1 = np.random.uniform(0, 1000, n_samples)
feature_2 = np.random.uniform(0, 1, n_samples)

X = np.column_stack([feature_1, feature_2])
y = (3 * feature_1 + 50 * feature_2 + 10).reshape(-1, 1)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std  # تعديل: توحيد مقياس المدخلات لأن الأعمدة كانت بمقاييس مختلفة

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std  # تعديل: توحيد مقياس الهدف لتسهيل استقرار التدريب

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)  # صحيح: لدينا ميزتان في X، لذلك عدد المدخلات 2
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # تعديل: lr=1.0 كان مرتفعًا جدًا وقد يسبب عدم استقرار التدريب

epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

print("\n--- انتهى التدريب ---")
print(f"Loss النهائي: {loss.item():.6f}")
