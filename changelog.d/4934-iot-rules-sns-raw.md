### Fixed: the e-stop and fault rules deliver their records to SNS

`rules.rules_for(..., sns_topic_arn=...)` built its SNS actions with `messageFormat: "JSON"`, which makes SNS refuse
any body without a `"default"` key. Every rule record is such a body, so no e-stop or high-severity fault ever reached
the topic, and with no error action on the rule nothing reported it. Both SNS actions now send the record as the raw
message body: a fleet e-stop reached an SQS subscriber of the topic 0.15 s after the broadcast.
