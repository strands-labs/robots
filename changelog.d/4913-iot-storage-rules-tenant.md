### Fixed: IoT storage rules file records under the publishing tenant and thing

The events and fault rules take the tenant and device from the topic, and the Basic Ingest telemetry rule takes
the tenant from the rule and the device from the MQTT client id. A `tenant` or `device_id` in the message body no
longer chooses the S3 key or the Timestream dimensions. Re-apply each tenant's rules with `apply_rules`; when
tenants share a bucket, scope each rule role's `s3:PutObject` to `recordings/<tenant>/*` (see docs/iot.md).
