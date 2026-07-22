
TEST_SPLIT_PRESETS = {
    'aug1': [
        'T1078-003/sequence_1_validaccount_pam-55_videoserver.json',
        'T1078-003/sequence_1_validaccount_sshkey-55_videoserver.json',
        'T1078-003/sequence_4-19_reposerver.json',
        'T1078-003/sequence_4-20_reposerver.json',
        'T1078-003/sequence_4-5_reposerver.json',
        'T1105-000/sequence_2_rootkit-23_videoserver.json',
        'T1105-000/sequence_3_ssh_apt-25_linuxshare.json',
        'T1105-000/sequence_3_ssh_healthcheck-25_linuxshare.json',
        'T1105-000/sequence_3_vnc_apt-25_linuxshare.json',
        'T1105-000/sequence_3_vnc_healthcheck-25_linuxshare.json',
        'T1105-000/sequence_3_vnc_puppet-25_linuxshare.json',
        'T1105-000/sequence_3_vnc_puppet-60_linuxshare.json',
        'T1133-000/sequence_7-3_docker.json',
        'T1205-001/sequence_4-9_inetfw.json',
        'T1219-000/sequence_6_screensharing_cron-45_client.json',
    ],
    'aug2': [
        'T1078-002/sequence_7-3_docker.json',
        'T1078-003/sequence_3_ssh_healthcheck-5_reposerver.json',
        'T1078-003/sequence_3_ssh_puppet-37_reposerver.json',
        'T1078-003/sequence_3_ssh_puppet-5_reposerver.json',
        'T1078-003/sequence_4-4_reposerver.json',
        'T1078-003/sequence_4-5_reposerver.json',
        'T1105-000/sequence_1_racecondition_localaccount-21_videoserver.json',
        'T1105-000/sequence_1_racecondition_pam-29_videoserver.json',
        'T1105-000/sequence_1_racecondition_sshkey-30_videoserver.json',
        'T1105-000/sequence_1_sudo_localaccount-38_videoserver.json',
        'T1105-000/sequence_1_validaccount_localaccount-38_videoserver.json',
        'T1105-000/sequence_2_rootkit-23_videoserver.json',
        'T1105-000/sequence_6_macro_binary-1_client.json',
        'T1219-000/sequence_6_screensharing_binary-57_client.json',
        'T1219-000/sequence_6_screensharing_cron-46_client.json',
    ],
    'aug3': [
        'T1071-001/sequence_2_cron-2_videoserver.json',
        'T1071-001/sequence_3_ssh_apt-16_reposerver.json',
        'T1078-003/sequence_1_validaccount_localaccount-56_videoserver.json',
        'T1078-003/sequence_1_validaccount_pam-56_videoserver.json',
        'T1078-003/sequence_3_ssh_puppet-37_reposerver.json',
        'T1078-003/sequence_4-19_reposerver.json',
        'T1078-003/sequence_4-4_reposerver.json',
        'T1105-000/sequence_1_cron_pam-29_videoserver.json',
        'T1105-000/sequence_1_sudo_sshkey-43_videoserver.json',
        'T1105-000/sequence_1_validaccount_sshkey-43_videoserver.json',
        'T1105-000/sequence_2_cron-19_videoserver.json',
        'T1105-000/sequence_3_vnc_puppet-60_linuxshare.json',
        'T1105-000/sequence_6_macro_binary-1_client.json',
        'T1219-000/sequence_6_screensharing_binary-53_client.json',
        'T1219-000/sequence_6_screensharing_binary-57_client.json',
    ],
    # Plain random stratified 20% split (seed=42) — same result as running
    # with no --test-file at all, just given a name so it can be selected
    # and compared alongside aug1/aug2/aug3 the same way. Not curated for
    # variant/unique balance like the aug* presets — a genuine random
    # baseline split.
    'random20': [
        'T1078-002/sequence_3_ssh_apt-1_reposerver.json',
        'T1078-003/sequence_3_ssh_puppet-37_reposerver.json',
        'T1078-003/sequence_3_ssh_puppet-5_reposerver.json',
        'T1078-003/sequence_4-19_reposerver.json',
        'T1105-000/sequence_1_pwnkit_localaccount-38_videoserver.json',
        'T1105-000/sequence_1_pwnkit_pam-41_videoserver.json',
        'T1105-000/sequence_1_sudo_pam-41_videoserver.json',
        'T1105-000/sequence_2_rootkit-3_videoserver.json',
        'T1105-000/sequence_3_ssh_puppet-25_linuxshare.json',
        'T1105-000/sequence_3_ssh_puppet-40_linuxshare.json',
        'T1105-000/sequence_3_vnc_apt-25_linuxshare.json',
        'T1105-000/sequence_3_vnc_puppet-25_linuxshare.json',
        'T1105-000/sequence_6_screensharing_cron-49_client.json',
        'T1133-000/sequence_3_ssh_healthcheck-1_reposerver.json',
        'T1133-000/sequence_7-3_docker.json',
        'T1190-000/sequence_1_cron_pam-5_videoserver.json',
        'T1190-000/sequence_1_pwnkit_localaccount-5_videoserver.json',
        'T1190-000/sequence_1_pwnkit_sshkey-5_videoserver.json',
        'T1205-001/sequence_4-9_inetfw.json',
        'T1219-000/sequence_6_screensharing_binary-41_client.json',
        'T1219-000/sequence_6_screensharing_binary-59_client.json',
        'T1219-000/sequence_6_screensharing_cron-49_client.json',
    ],
}


def resolve_test_file_arg(value):
    if not value:
        return None
    if value in TEST_SPLIT_PRESETS:
        return TEST_SPLIT_PRESETS[value]
    return [s.strip() for s in value.split(',')]
