# GitHub OAuth Setup Guide

This guide will help you set up GitHub OAuth authentication for the Codebase Analytics application.

## 🔧 Prerequisites

- A GitHub account
- Access to create GitHub OAuth Apps

## 📝 Step 1: Create a GitHub OAuth App

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **"New OAuth App"**
3. Fill in the application details:
   - **Application name**: `Codebase Analytics`
   - **Homepage URL**: `http://localhost:9996`
   - **Authorization callback URL**: `http://localhost:9996/api/auth/callback/github`
4. Click **"Register application"**

## 🔑 Step 2: Get Your Credentials

After creating the app, you'll see:
- **Client ID** - Copy this value
- **Client Secret** - Click "Generate a new client secret" and copy the value

## ⚙️ Step 3: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cd frontend
   cp .env.example .env.local
   ```

2. Edit `.env.local` and add your GitHub OAuth credentials:
   ```env
   # GitHub OAuth Configuration
   GITHUB_CLIENT_ID=your_actual_client_id_here
   GITHUB_CLIENT_SECRET=your_actual_client_secret_here
   
   # NextAuth Configuration
   NEXTAUTH_URL=http://localhost:9996
   NEXTAUTH_SECRET=your_random_secret_here
   
   # Backend API
   NEXT_PUBLIC_API_URL=http://localhost:9997
   ```

3. Generate a random secret for `NEXTAUTH_SECRET`:
   ```bash
   # You can use this command to generate a random secret:
   openssl rand -base64 32
   ```

## 🚀 Step 4: Launch the Application

1. **Start the backend** (Terminal 1):
   ```bash
   cd backend
   python api.py
   ```

2. **Start the frontend** (Terminal 2):
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**:
   - Open http://localhost:9996 in your browser
   - Click "Sign in with GitHub"
   - Authorize the application
   - Select repositories from the dropdown

## 🔍 Features Available After Setup

✅ **GitHub Authentication**: Secure OAuth login with GitHub
✅ **Repository Selection**: Dropdown with all your accessible repositories
✅ **Repository Analysis**: Analyze any selected repository
✅ **Private Repository Support**: Access to both public and private repos
✅ **Repository Metadata**: View stars, forks, language, and description

## 🛠️ Troubleshooting

### Common Issues:

1. **"Invalid client" error**:
   - Check that your `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` are correct
   - Ensure the callback URL in GitHub matches exactly: `http://localhost:9996/api/auth/callback/github`

2. **"NEXTAUTH_SECRET" error**:
   - Make sure you've set a `NEXTAUTH_SECRET` in your `.env.local` file
   - The secret should be at least 32 characters long

3. **Repository list not loading**:
   - Check that the backend is running on port 9997
   - Verify the GitHub token has the correct scopes (read:user, user:email, repo)

4. **CORS errors**:
   - Ensure both frontend (9996) and backend (9997) are running
   - Check that the API proxy in `next.config.js` is correctly configured

## 🔐 Security Notes

- Never commit your `.env.local` file to version control
- Keep your GitHub Client Secret secure
- The `NEXTAUTH_SECRET` should be unique and random
- For production, use HTTPS URLs instead of HTTP

## 📚 Additional Resources

- [NextAuth.js Documentation](https://next-auth.js.org/)
- [GitHub OAuth Apps Documentation](https://docs.github.com/en/developers/apps/building-oauth-apps)
- [Octokit REST API Documentation](https://octokit.github.io/rest.js/)

