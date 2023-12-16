class S3Path(AnyUrl):
    allowed_schemes = {"s3"}
    user_required = False
    max_length = 1023
    min_length = 8

    @field_validator("url")
    @classmethod
    def validate_s3_url(cls, v):
        if not v.startswith("s3://"):
            raise ValueError("S3 path must start with s3://")
        if len(v) < cls.min_length:
            raise ValueError("S3 path is too short")
        if len(v) > cls.max_length:
            raise ValueError("S3 path is too long")
        return v

    @field_validator("host")
    @classmethod
    def validate_bucket_name(cls, v):
        if not v:
            raise ValueError("Bucket name cannot be empty")
        if not 3 <= len(v) <= 63:
            raise ValueError("Bucket name must be between 3 and 63 characters")
        if not re.match(r"^[a-z0-9.\-]+$", v):
            raise ValueError(
                "Bucket name can only contain lowercase letters, numbers, dots, and hyphens"
            )
        if v.startswith("-") or v.endswith("-"):
            raise ValueError("Bucket name cannot start or end with a hyphen")
        if ".." in v:
            raise ValueError("Bucket name cannot have consecutive periods")
        return v

    @field_validator("path")
    @classmethod
    def validate_key(cls, v):
        if "//" in v:
            raise ValueError("Key must not contain double slashes")
        if "\\" in v:
            raise ValueError("Backslashes are not standard in S3 paths")
        if "\t" in v or "\n" in v:
            raise ValueError("Key cannot contain tab or newline characters")
        return v.strip("/")
