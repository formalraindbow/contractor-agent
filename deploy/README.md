# Stand on a VPS

The production stack runs one API process and Caddy. Only ports 80/443 are
published; the MCP tools run inside the API process. Caddy obtains and renews
HTTPS certificates, and Docker restarts both services after a reboot.

## First deployment

1. Install Docker Engine and the Compose plugin from Docker's official packages.
2. Point a domain at the VPS public IPv4 address. A name such as
   `kontragent-203-0-113-10.sslip.io` can also be used for a demo: replace the
   example address with the actual public IP and verify DNS before deploying.
3. Copy the source to `/opt/contractor-agent`, without local secrets, caches or
   virtual environments. Copy `deploy/production.env.example` to
   `.env.production`, fill the host, model and API key, then `chmod 600` it.
   Keep the same `WEB_PASSWORD` if existing stand links should remain usable.
4. Prepare persistent directories:

   ```sh
   install -d -m 750 -o 10001 -g 10001 state runs
   install -d -m 700 backups
   ```

5. Before first launch, migrate the previous stand's checkpoint database using
   `python deploy/backup_sessions.py SOURCE_DB BACKUP_DIR`. Copy the resulting
   consistent backup to `state/sessions.sqlite` and set ownership to
   `10001:10001`. Never replace a database under a running API. Keep the source
   database and an untouched copy of the backup until verification completes.
6. Start the stack:

   ```sh
   docker compose --env-file .env.production -f docker-compose.production.yml up -d --build
   docker compose --env-file .env.production -f docker-compose.production.yml ps
   ```

7. Verify HTTPS, password entry, an existing conversation, a new streamed answer,
   and the conversation after an API restart. Do not print the full environment
   or API keys to terminal logs.

## Backups and updates

Run the following daily with cron or a systemd timer. The backup uses SQLite's
online backup API, so it includes WAL transactions without stopping the agent:

```sh
python3 /opt/contractor-agent/deploy/backup_sessions.py \
  /opt/contractor-agent/state/sessions.sqlite /opt/contractor-agent/backups
```

The provided systemd units can be installed with:

```sh
cp deploy/kontragent-backup.service deploy/kontragent-backup.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now kontragent-backup.timer
systemctl start kontragent-backup.service
```

Keep a separate copy away from the VPS. Backups in the same VPS survive an
application restart but not deletion of the VPS or a disk failure.

For updates, preserve `.env.production`, `state/`, `runs/`, `backups/` and the
Caddy volumes. Back up sessions, replace source files, then run the same
`up -d --build` command. Never use `down -v`: it deletes the certificate volumes.
