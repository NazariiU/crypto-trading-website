# Cryptocurrency Trading Web Application - Deployment Guide

## Introduction

This guide provides instructions for deploying the Cryptocurrency Trading Web Application to Streamlit Cloud for permanent access. Streamlit Cloud offers free hosting for Streamlit applications with GitHub integration.

## Prerequisites

Before deploying the application, ensure you have:

1. A GitHub account
2. Git installed on your local machine
3. The complete cryptocurrency trading web application code

## Deployment Steps

### 1. Create a GitHub Repository

1. Log in to your GitHub account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "crypto-trading-website")
4. Choose visibility (public or private)
5. Click "Create repository"

### 2. Push Your Code to GitHub

1. Initialize a Git repository in your local project folder:
   ```bash
   cd /path/to/crypto_trading_website
   git init
   ```

2. Add all files to the repository:
   ```bash
   git add .
   ```

3. Commit the changes:
   ```bash
   git commit -m "Initial commit of Cryptocurrency Trading Web Application"
   ```

4. Add your GitHub repository as a remote:
   ```bash
   git remote add origin https://github.com/yourusername/crypto-trading-website.git
   ```

5. Push your code to GitHub:
   ```bash
   git push -u origin main
   ```

### 3. Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and the main file (`streamlit_app.py`)
5. Click "Deploy"

### 4. Configure Environment Variables

For security, you should set environment variables in Streamlit Cloud:

1. In your Streamlit Cloud dashboard, select your app
2. Go to "Settings" > "Secrets"
3. Add the following secrets:
   - `ENCRYPTION_KEY`: A secure encryption key for API secrets
   - `ENCRYPTION_SALT`: A secure salt for encryption
   - `DB_TYPE`: Database type (sqlite or postgres)
   - Any other sensitive configuration values

### 5. Verify Deployment

1. Once deployment is complete, Streamlit Cloud will provide a URL for your application
2. Visit the URL to ensure the application is working correctly
3. Test all functionality including:
   - User registration and login
   - Market data visualization
   - Strategy backtesting
   - Settings configuration

### 6. Set Up Custom Domain (Optional)

If you want to use a custom domain:

1. Purchase a domain from a domain registrar
2. In your Streamlit Cloud dashboard, go to "Settings" > "Custom domain"
3. Follow the instructions to configure your domain

## Maintenance

### Updating Your Application

To update your deployed application:

1. Make changes to your local code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push
   ```
3. Streamlit Cloud will automatically redeploy your application

### Monitoring

The application includes built-in monitoring:

1. Health checks run automatically to verify system status
2. Performance metrics are collected for analysis
3. Automated backups are performed according to the configured schedule

## Troubleshooting

### Common Deployment Issues

1. **Missing Dependencies**:
   - Ensure all required packages are listed in `requirements.txt`
   - Check Streamlit Cloud logs for import errors

2. **Environment Variables**:
   - Verify all required secrets are set in Streamlit Cloud
   - Check for typos in secret names

3. **Database Connection Issues**:
   - For SQLite, ensure the database path is correct
   - For PostgreSQL, verify connection parameters

### Streamlit Cloud Limitations

Be aware of Streamlit Cloud's limitations:

1. Free tier has resource constraints
2. Applications may sleep after periods of inactivity
3. File system changes are not persistent

## Support

If you encounter issues with deployment, contact:
- Streamlit support: https://support.streamlit.io/
- Application support: support@example.com

## Security Considerations

1. Never commit sensitive information (API keys, passwords) to your repository
2. Use environment variables for all sensitive configuration
3. Regularly update dependencies to patch security vulnerabilities
4. Enable two-factor authentication on your GitHub account

## Conclusion

Your Cryptocurrency Trading Web Application is now deployed and accessible to users worldwide. The application will automatically update whenever you push changes to your GitHub repository.
