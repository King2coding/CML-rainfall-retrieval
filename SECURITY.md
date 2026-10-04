# Security

Do not commit credentials, populated `.env` files, private endpoints, or
restricted research data. Use environment variables or local configuration
excluded by `.gitignore`.

If a credential is committed accidentally, remove it from active use, rotate
it with the service provider, and sanitize the repository history before making
the repository public.
