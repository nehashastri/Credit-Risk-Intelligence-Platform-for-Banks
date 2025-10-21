# 🏦 Credit Risk Intelligence Platform (Phase 1)

A comprehensive Streamlit web application for credit risk analysis, integrating BigQuery data with real-time market and news sentiment analysis.

## 📊 Overview

The Credit Risk Intelligence Platform provides a unified dashboard for monitoring credit risk through multiple data sources:

- **BigQuery Integration**: 4 landing tables with macro and news data
- **Market Analysis**: Real-time ETF data from 7 sector/index funds
- **News Sentiment**: AI-powered sentiment analysis across 7 economic topics
- **Risk Assessment**: Composite Macro Risk Index (CMRI) calculation

## 🚀 Features

### 📋 Pages

1. **🏠 Home**: Platform overview, data freshness, and connection status
2. **📊 Data Overview**: Row counts, missing data analysis, timestamps, macro indicators overview, and news articles analysis
3. **🔍 Exploratory Data Analysis**: Comprehensive analysis combining macro trends, market performance, news sentiment, and risk monitoring

### 🔧 Key Components

- **Modular Architecture**: Clean separation of concerns with utils modules
- **Real-time Data**: Live market data via yfinance integration
- **Interactive Visualizations**: Plotly charts with drill-down capabilities
- **Risk Monitoring**: Automated alerts and trend analysis
- **Data Export**: CSV export functionality for all analyses

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Google Cloud Platform account with BigQuery access
- Service account credentials for BigQuery

### 1. Clone Repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd StreamlitPhase1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Credentials

Update `.streamlit/secrets.toml` with your actual BigQuery credentials:

```toml
[gcp]
project_id = "your-project-id"
dataset_id = "your-dataset"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "your-private-key"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
# ... other credentials
```

### 4. Run Application

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
StreamlitPhase1/
├── Home.py                    # Main application entry point
├── pages/                     # Page modules
│   ├── data_overview.py      # Data overview and quality analysis
│   ├── macro_eda.py          # Macroeconomic analysis
│   ├── market_analysis.py    # Market performance analysis
│   ├── news_sentiment.py    # News sentiment analysis
│   └── risk_dashboard.py     # Risk monitoring dashboard
├── utils/                    # Utility modules
│   ├── gcp_connect.py       # BigQuery connection handler
│   ├── charts.py            # Chart building utilities
│   └── cmri.py              # Composite Macro Risk Index calculator
├── .streamlit/
│   └── secrets.toml         # Configuration and credentials
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .github/
    └── workflows/
        └── deploy.yml        # GitHub Actions deployment
```

## 🔗 Data Sources

### BigQuery Tables

- `fact_credit_outcomes`: Credit risk outcomes and metrics
- `fact_macro_indicators_daily`: Daily macroeconomic indicators
- `fact_macro_indicators_monthly`: Monthly macroeconomic indicators
- `news_articles`: Raw news articles with sentiment scores and metadata

### Market Data (yfinance)

- **SPY**: S&P 500 ETF
- **XLF**: Financial Sector ETF
- **XLY**: Consumer Discretionary ETF
- **XLK**: Technology Sector ETF
- **XLE**: Energy Sector ETF
- **XLI**: Industrial Sector ETF
- **XLRE**: Real Estate Sector ETF

### News Topics

- **fed_policy**: Federal Reserve policy and interest rates
- **labor**: Employment and job market trends
- **markets**: Stock market and financial markets
- **energy**: Oil prices and energy sector
- **real_estate**: Housing market and property trends
- **cpi**: Consumer Price Index and inflation
- **layoffs**: Corporate layoffs and job cuts

## 📊 Composite Macro Risk Index (CMRI)

The CMRI combines three risk components:

- **Macro Risk (40%)**: Unemployment, CPI, Fed funds rate, GDP
- **Market Risk (35%)**: ETF volatility and returns
- **News Risk (25%)**: Sentiment scores across topics

### Risk Levels

- 🟢 **Low Risk**: CMRI < 0.3
- 🟡 **Medium Risk**: CMRI 0.3 - 0.6
- 🔴 **High Risk**: CMRI > 0.6

## 🚀 Deployment

### Local Development

```bash
streamlit run Home.py --server.port 8501
```

### GitHub Actions Deployment

The repository includes a GitHub Actions workflow for automated deployment to Streamlit Cloud.

### Docker Deployment

```bash
docker build -t credit-risk-platform .
docker run -p 8501:8501 credit-risk-platform
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON
- `STREAMLIT_SERVER_PORT`: Port for Streamlit server
- `STREAMLIT_SERVER_ADDRESS`: Server address

### Customization

- **Risk Weights**: Modify weights in `utils/cmri.py`
- **Chart Styling**: Update colors and templates in `utils/charts.py`
- **Data Sources**: Add new tables in `utils/gcp_connect.py`

## 📈 Usage Examples

### Loading Data

1. Navigate to any analysis page
2. Select date range or time period
3. Click "Load Data" button
4. View interactive charts and metrics

### Exporting Results

1. Run analysis on any page
2. Click "Export" buttons for CSV downloads
3. Use exported data for external analysis

### Risk Monitoring

1. Visit Risk Dashboard
2. Monitor CMRI trends and alerts
3. Review risk component breakdown
4. Generate forecasts for planning

## 🐛 Troubleshooting

### Common Issues

**BigQuery Connection Error**
- Verify service account credentials
- Check project ID and dataset permissions
- Ensure billing is enabled

**Missing Data**
- Check date ranges for data availability
- Verify table names in BigQuery
- Review data freshness timestamps

**Chart Display Issues**
- Clear browser cache
- Check Plotly version compatibility
- Verify data format and types

### Debug Mode

Run with debug logging:

```bash
streamlit run Home.py --logger.level debug
```

## 📝 TODO Items

- [ ] Add GitHub repository URL to configuration
- [ ] Implement custom visualization templates
- [ ] Add more sophisticated forecasting models
- [ ] Integrate additional data sources
- [ ] Add user authentication and access control
- [ ] Implement real-time data streaming
- [ ] Add mobile-responsive design improvements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:

- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the configuration documentation

## 🔄 Version History

- **v1.0.0**: Initial release with Phase 1 features
  - BigQuery integration
  - Market analysis
  - News sentiment
  - Risk dashboard
  - CMRI calculation

---

**🏦 Credit Risk Intelligence Platform | Phase 1 | Built with Streamlit**

*Ready to run with `streamlit run Home.py`*
