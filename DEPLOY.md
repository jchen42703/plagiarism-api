# Deploy

# Docker

```
cd earthquake_forecasting
docker-compose up
```

## Install nginx

```
sudo apt update
sudo apt install nginx
```

```
sudo ufw app list
sudo ufw allow 'Nginx HTTP'
sudo ufw allow 'Nginx HTTPS'
sudo ufw status
```

## Configure nginx

```
nano /etc/nginx/sites-available/default
```

```
server {
    server_name earthquakedamageforecast.com www.earthquakedamageforecast.com;
    listen 80;

    location /api/ {
        proxy_pass http://localhost:5001/api/;
        proxy_http_version 1.1;
    }
    location / {
        proxy_pass http://localhost:3006;
    }
}
```

To test:

```
sudo nginx -t
```

```
sudo systemctl restart nginx
```

**To set up SSL:**

Install certbot:

```
sudo apt-get install certbot
apt-get install python3-certbot-nginx
```

(Test)

```
sudo certbot --nginx --staging -d earthquakedamageforecast.com -d www.earthquakedamageforecast.com
```

(Real)

```
sudo certbot --nginx -d earthquakedamageforecast.com -d www.earthquakedamageforecast.com
```

SSL Crontab:

```
crontab -e

// add this in crontab file; renews if cert will expire in the next 30 days
0 12 * * * /usr/bin/certbot renew --quiet
```
