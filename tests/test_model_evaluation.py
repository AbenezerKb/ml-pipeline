import pytest
import numpy as np
from unittest.mock import patch
from pipelines.model_evaluation import evaluate_model, plot_confusion_matrix


def test_evaluate_model_returns_correct_metrics():
    """Test that evaluate_model returns the correct metrics"""
    # example data
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8])
    
    result = evaluate_model("Test Model", y_true, y_pred, y_prob)
    
    # Check that all expected keys are present
    assert 'Model' in result
    assert 'Accuracy' in result
    assert 'Precision' in result
    assert 'Recall' in result
    assert 'F1' in result
    assert 'AUC' in result
    
    # Check that the model name is correct
    assert result['Model'] == "Test Model"
    
    #check that metrics are within valid ranges
    assert 0 <= result['Accuracy'] <= 1
    assert 0 <= result['Precision'] <= 1
    assert 0 <= result['Recall'] <= 1
    assert 0 <= result['F1'] <= 1
    assert 0 <= result['AUC'] <= 1


def test_evaluate_model_perfect_predictions():
    """Test with perfect predictions"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    
    result = evaluate_model("Perfect Model", y_true, y_pred, y_prob)
    
    # Check that all metrics are 1.0
    assert result['Accuracy'] == pytest.approx(1.0)
    assert result['Precision'] == pytest.approx(1.0)
    assert result['Recall'] == pytest.approx(1.0)
    assert result['F1'] == pytest.approx(1.0)
    assert result['AUC'] == pytest.approx(1.0)


def test_evaluate_model_prints_results(capsys):
    """Test that evaluate_model correctly displays the results"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    evaluate_model("Test Model", y_true, y_pred, y_prob)
    
    # Capture the printed output
    captured = capsys.readouterr()
    
    # Check that the output contains expected strings
    assert "Test Model" in captured.out
    assert "Accuracy" in captured.out
    assert "Precision" in captured.out
    assert "Recall" in captured.out
    assert "F1-Score" in captured.out
    assert "ROC-AUC" in captured.out


@patch('pipelines.model_evaluation.plt.show')
@patch('pipelines.model_evaluation.sns.heatmap')
def test_plot_confusion_matrix_called_correctly(mock_heatmap, mock_show):
    """Test que plot_confusion_matrix appelle les bonnes fonctions"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    
    plot_confusion_matrix(y_true, y_pred, "Test Model")
    
    #Check that heatmap was called
    assert mock_heatmap.called
    
    # Check that plt.show was called
    assert mock_show.called


@patch('pipelines.model_evaluation.plt.show')
def test_plot_confusion_matrix_with_perfect_predictions(mock_show):
    """Test of the confusion matrix with perfect predictions"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    plot_confusion_matrix(y_true, y_pred, "Perfect Model")
    
    assert mock_show.called