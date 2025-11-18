# Credit-Risk-Intelligence-Platform-for-Banks

**Introduction:**

Credit card portfolios are highly sensitive to shifts in macroeconomic conditions such as interest rates, inflation, employment, and GDP growth. Yet, most financial institutions still assess portfolio risk largely in isolation from the broader economy. This siloed view limits their ability to anticipate early warning signals of consumer distress and to prepare for regime changes, such as the transition from economic expansion to recession. Traditional credit risk models, often focused on borrower-level behavior, fail to capture how external macroeconomic shocks ripple through consumer credit performance. As a result, banks struggle to act proactively in adjusting underwriting strategies, provisioning for losses, and optimizing capital reserves.
Our project aims to bridge this gap by integrating macroeconomic, market, and news-based data into a unified analytics pipeline. This pipeline will generate actionable insights, predictive signals, and scenario-based simulations to help financial institutions manage risk more intelligently and respond dynamically to changing economic conditions.

**Business Proposal:**

The proposal centers on developing an Early Warning System for Credit Risk that leverages macroeconomic and market indicators to predict credit card delinquency rates and support proactive risk management. It also includes Regulatory Stress Testing and Scenario Simulation to quantify potential credit losses under economic shocks such as unemployment spikes or GDP declines. Additionally, a Policy Impact Analysis component will assess how monetary or fiscal policy changes, like interest rate hikes or inflation surges, affect delinquency and loss patterns. Finally, a GenAI-Powered Risk Advisor will provide an interactive interface enabling business users to conduct “what-if” analyses, explore scenario outcomes, and receive contextual explanations through a conversational AI tool. This multi-pronged approach transforms the project from a static forecasting tool into a dynamic support system, integrating prediction, diagnosis, and communication for decision-making.

**Data Overview:**

Our pipeline integrates three complementary data sources to provide a complete view of the financial environment influencing credit card performance:
- FRED (Federal Reserve Economic Data): Provides authoritative macroeconomic indicators, such as GDP, unemployment, inflation, interest rates, and consumer credit, updated daily to quarterly. These serve as leading indicators of consumer financial health, helping detect early signs of stress before delinquency rates rise.
- yFinance: Supplies daily financial market and sector-level data (eg, S&P 500, sector ETFs) that capture investor sentiment, liquidity conditions, and market expectations. This data complements FRED by offering a forward-looking view of economic sentiment, translating market dynamics into a macroeconomic risk context.
- Tavily API: Delivers real-time textual data on economic events, policy announcements, and financial news sentiment, enriching the dataset with qualitative context behind quantitative trends. This enables the pipeline to understand how narrative sentiment may amplify or mitigate observed economic effects.
Together, these datasets create a comprehensive ecosystem where FRED anchors the pipeline with structured economic fundamentals, yFinance injects market-driven perspectives, and Tavily provides contextual intelligence from unstructured text data.
