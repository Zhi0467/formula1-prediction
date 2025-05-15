# Formula 1 Race Prediction System

A modular and extensible system for predicting Formula 1 race outcomes. This system ingests various data sources, engineers features, utilizes multiple predictive models, and combines their outputs using ensemble methods to produce a final predicted race ranking.

## Project Overview

The F1 prediction system follows a modular design with the following components:

1. **Data Ingestion & Sources**
   - Ergast/Jolpica API
   - FastF1 API
   - OpenF1 API
   - Weather data
   - Polymarket odds
   - Manual data inputs

2. **Feature Engineering Hub**
   - Driver, team, and track statistics
   - Historical performance
   - Weather features
   - Domain knowledge encoding

3. **Predictive Models**
   - Lap Time Prediction Model
   - Rank Delta Prediction Model
   - Head-to-Head Comparison Model
   - (Optional) Text Analysis Model

4. **Ensemble & Ranking Aggregation**
   - Weighted averaging
   - Borda count
   - Learning-to-Rank aggregation

## Setup

### Prerequisites

- Python 3.8 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scipy
  - requests
  - xgboost
  - lightgbm
  - pyyaml

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/formula1-prediction.git
   cd formula1-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the system by editing `f1_predictor/config/config.yaml`

## Usage

### Making Predictions

To generate race predictions, use the `run_predict.py` script:

```
python -m f1_predictor.run_predict --race <race_name_or_round> --season <year>
```

Example:
```
python -m f1_predictor.run_predict --race "Bahrain Grand Prix" --season 2024
```

Additional options:
- `--config <path>`: Use a custom configuration file
- `--expert-input <path>`: Incorporate expert predictions from a CSV file
- `--output <path>`: Save predictions to a file
- `--verbose`: Enable verbose logging

### Training Models

To train the prediction models on historical data:

```
python -m f1_predictor.run_training --seasons <start_year>-<end_year>
```

Example:
```
python -m f1_predictor.run_training --seasons 2021-2023
```

## Extending the System

The system is designed for extensibility:

1. **Adding New Data Sources**: Create a new client in `data_ingest/` and update the configuration.

2. **Adding New Features**: Add functions to the relevant feature module and update the configuration.

3. **Adding New Models**: Create a new model class inheriting from `BaseModel` and update the configuration.

## Project Structure

```
f1_predictor/
├── data_ingest/                # Data fetching modules
├── features/                   # Feature engineering modules
├── models/                     # Prediction model implementations
├── ensemble/                   # Ensemble and ranking aggregation
├── utils/                      # Utility functions
├── config/                     # Configuration files
├── data/                       # Raw and processed data
│   ├── raw/                    
│   └── processed/              
├── trained_models/             # Saved model files
├── run_predict.py              # CLI for predictions
├── run_training.py             # CLI for model training
```

## Configuration

The system is controlled via the `config.yaml` file, which allows customization of:

- Data sources
- Feature engineering parameters
- Model selection and hyperparameters
- Ensemble methods and weights
- Evaluation metrics

## License

[MIT License](LICENSE)

## Acknowledgments

- [Ergast F1 API](http://ergast.com/mrd/)
- [FastF1](https://github.com/theOehrly/Fast-F1)
- [OpenF1](https://openf1.org/)