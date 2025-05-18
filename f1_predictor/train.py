from models.base_model import AbstractBasePredictor
from models.final_pos_model import FinalPositionPredictor

DATA_DIR_GLOBAL = "test_output/dataset"

if __name__ == "__main__":
    custom_xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,      
        'learning_rate': 5e-1,    
        'max_depth': 6,           
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 123,
        'early_stopping_rounds': 25, 
        'gamma': 0.01,
        'reg_alpha': 0.2,
        'reg_lambda': 1.8,
        'eval_metric': ["rmse"]    
    }

    # predictor = FinalPositionPredictor(data_dir=DATA_DIR_GLOBAL, model_params=custom_xgb_params)
    predictor = FinalPositionPredictor(data_dir=DATA_DIR_GLOBAL) 
    
    try:
        results = predictor.run_pipeline()
        if results:
             print("\nFinal Test Metrics:", results)
    except FileNotFoundError as e:
        print(f"Halting due to error: {e}")
    except ValueError as e:
        print(f"Halting due to error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()