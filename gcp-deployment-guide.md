# Deploying Your WebSocket Application to Google Cloud Platform

This guide will walk you through the process of deploying your WebSocket application to a Google Cloud Platform (GCP) virtual machine. We'll be moving your `websocket_server.py` and `index.html` files from a local setup to a cloud-based environment. This will allow users to access your application without the need for ngrok.

## Step 1: Set up a Google Cloud Platform Virtual Machine

1. Log in to the Google Cloud Console (console.cloud.google.com).
2. Create a new project or select an existing one.
3. Navigate to Compute Engine > VM instances.
4. Click "Create Instance".
5. Choose a name for your instance.
6. Select a region and zone close to your target users.
7. Choose a machine type (e.g., e2-medium).
8. Select a boot disk (e.g., Ubuntu 20.04 LTS).
9. In the Firewall section, check "Allow HTTP traffic" and "Allow HTTPS traffic".
10. Click "Create" to launch your VM.

## Step 2: Connect to Your VM and Set Up the Environment

1. In the VM instances list, click the "SSH" button next to your new instance.
2. Once connected, update the package list:
   ```
   sudo apt-get update
   ```
3. Install Python and pip if they're not already installed:
   ```
   sudo apt-get install python3 python3-pip
   ```
4. Install any required Python packages for your `websocket_server.py`:
   ```
   pip3 install websockets asyncio pandas
   ```
   (Add any other packages your script needs)

## Step 3: Transfer Your Files to the VM

1. In the Cloud Console, click on your VM instance.
2. In the SSH terminal, create a new directory for your application:
   ```
   mkdir myapp
   cd myapp
   ```
3. Use the built-in file editor or SCP to transfer `websocket_server.py` and `index.html` to this directory.

## Step 4: Modify Your WebSocket Server

1. Open `websocket_server.py` in an editor:
   ```
   nano websocket_server.py
   ```
2. Modify the `host` variable to listen on all interfaces:
   ```python
   host = '0.0.0.0'
   ```
3. Ensure the port is set to 8080 (or your preferred port):
   ```python
   port = 8080
   ```
4. Save and exit the editor.

## Step 5: Set Up Nginx as a Reverse Proxy

1. Install Nginx:
   ```
   sudo apt-get install nginx
   ```
2. Create a new Nginx configuration file:
   ```
   sudo nano /etc/nginx/sites-available/myapp
   ```
3. Add the following configuration:
   ```nginx
   server {
       listen 80;
       server_name _;

       location / {
           root /home/your_username/myapp;
           index index.html;
       }

       location /ws {
           proxy_pass http://localhost:8080;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```
4. Create a symbolic link to enable the site:
   ```
   sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
   ```
5. Test the Nginx configuration:
   ```
   sudo nginx -t
   ```
6. If the test is successful, restart Nginx:
   ```
   sudo systemctl restart nginx
   ```

## Step 6: Update Your index.html

1. Open `index.html` in an editor:
   ```
   nano index.html
   ```
2. Modify the `wsUrl` to use the '/ws' path and the current hostname:
   ```javascript
   const wsUrl = `ws://${window.location.hostname}/ws`;
   ```
3. Save and exit the editor.

## Step 7: Set Up a Process Manager (PM2)

1. Install Node.js and npm:
   ```
   sudo apt-get install nodejs npm
   ```
2. Install PM2 globally:
   ```
   sudo npm install -g pm2
   ```
3. Start your WebSocket server with PM2:
   ```
   pm2 start websocket_server.py --name myapp --interpreter python3
   ```
4. Set PM2 to start on boot:
   ```
   pm2 startup systemd
   ```
   Follow the instructions it provides.
5. Save the current process list:
   ```
   pm2 save
   ```

## Step 8: Configure Firewall Rules

1. In the GCP Console, go to VPC Network > Firewall.
2. Click "Create Firewall Rule".
3. Name your rule (e.g., "allow-websocket").
4. Set the following:
   - Direction of traffic: Ingress
   - Action on match: Allow
   - Targets: All instances in the network
   - Source filter: IP ranges
   - Source IP ranges: 0.0.0.0/0
   - Protocols and ports: Specify protocols and ports
     - TCP: 8080 (or your chosen port)
5. Click "Create".

## Step 9: Access Your Application

Your application should now be accessible at `http://YOUR_VM_IP_ADDRESS`. Users can visit this URL to use your WebSocket application.

## Optional: Set Up HTTPS

For added security, you should consider setting up HTTPS. You can do this by:

1. Obtaining an SSL certificate (Let's Encrypt is a free option).
2. Configuring Nginx to use HTTPS.
3. Updating your `index.html` to use `wss://` instead of `ws://`.

This setup will give you a robust, cloud-based deployment of your WebSocket application. Users will be able to access it directly through the VM's IP address or a domain name if you set one up, without the need for ngrok.

Remember to keep your GCP instance and all software up to date for security purposes. Also, monitor your usage to manage costs effectively.
