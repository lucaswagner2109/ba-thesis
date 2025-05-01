import numpy as np
import torch

def train_model(model, t_loader, v_loader, optimizer, criterion, save_path, device, max_epochs=500, patience=20):
    ts, vs = [], []

    best_loss = np.inf
    best_model = None
    count = 0
    
    for e in range(max_epochs):
        # -- TRAINING --
        model.train()
        sum_t_loss = 0

        for X_b, y_b in t_loader:
            X_b, target = X_b.to(device), y_b.to(device)
            
            # prediction
            t_pred = model(X_b)
            t_loss = criterion(t_pred, target)
            sum_t_loss += t_loss.item()
            
            # update model
            optimizer.zero_grad()
            t_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # regulate gradients
            optimizer.step()
            
        # store avg loss
        avg_t_loss = sum_t_loss / len(t_loader)
        ts.append(avg_t_loss)

        # -- VALIDATION --
        model.eval()
        sum_v_loss = 0

        with torch.no_grad():
            for X_b, y_b in v_loader:
                X_b, target = X_b.to(device), y_b.to(device)

                # prediction
                v_pred = model(X_b)
                v_loss = criterion(v_pred, target)
                sum_v_loss += v_loss.item()

        # store avg loss
        avg_v_loss = sum_v_loss / len(v_loader)
        vs.append(avg_v_loss)

        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            best_model = model.state_dict()
            count = 0
        else:
            count += 1

        # stop early if no improvement
        if count > patience:
            print(f"Stopping at epoch {e+1}")
            torch.save(best_model, save_path)
            return ts, vs

        # Show training progress
        if e%10 == 0:
            print(f"Epoch {e+1}: Training Loss: {avg_t_loss:.4f}, Validation Loss: {avg_v_loss:.4f}")
            
    torch.save(best_model, save_path)
    return ts, vs