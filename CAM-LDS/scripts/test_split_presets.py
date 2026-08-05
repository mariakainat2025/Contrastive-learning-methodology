
TEST_SPLIT_PRESETS = {
    'scenario1': [
        '1_autostart_localaccount-10', '1_autostart_localaccount-12', '1_autostart_localaccount-15',
        '1_autostart_localaccount-16', '1_autostart_localaccount-5', '1_autostart_localaccount-7',
        '1_cron_localaccount-32', '1_cron_localaccount-33', '1_cron_pam-29',
        '1_pwnkit_localaccount-35', '1_pwnkit_localaccount-38', '1_pwnkit_localaccount-39',
        '1_pwnkit_pam-41', '1_pwnkit_pam-42', '1_pwnkit_sshkey-43', '1_pwnkit_sshkey-44',
        '1_racecondition_localaccount-46', '1_racecondition_localaccount-48',
        '1_racecondition_localaccount-49', '1_racecondition_localaccount-50',
        '1_racecondition_localaccount-52', '1_validaccount_localaccount-55',
        '1_validaccount_localaccount-56',
    ],
    'scenario2': [
        '2_cron-1', '2_cron-11', '2_cron-17', '2_cron-19', '2_cron-2', '2_cron-3', '2_cron-4',
        '2_cron-8', '2_rootkit-22', '2_rootkit-23', '2_rootkit-26', '2_rootkit-27', '2_rootkit-28',
        '2_rootkit-29', '2_rootkit-8',
    ],
    'scenario3': [
        '3_ssh_apt-1', '3_ssh_apt-10', '3_ssh_apt-12', '3_ssh_apt-14', '3_ssh_apt-16',
        '3_ssh_apt-17', '3_ssh_apt-18', '3_ssh_apt-19', '3_ssh_apt-2', '3_ssh_apt-20',
        '3_ssh_apt-23', '3_ssh_apt-25', '3_ssh_apt-3', '3_ssh_apt-4', '3_ssh_apt-5',
        '3_ssh_healthcheck-34', '3_ssh_puppet-37', '3_ssh_puppet-40', '3_ssh_puppet-6',
        '3_vnc_apt-43', '3_vnc_apt-45', '3_vnc_apt-49', '3_vnc_apt-51', '3_vnc_apt-54',
        '3_vnc_healthcheck-55', '3_vnc_puppet-58', '3_vnc_puppet-60',
    ],
    'scenario4': [
        '4-12', '4-13', '4-14', '4-15', '4-16', '4-19', '4-20', '4-21', '4-3', '4-4', '4-5',
        '4-6', '4-9',
    ],
    'scenario5': [
        '5-2', '5-3',
    ],
    'scenario6': [
        '6_macro_binary-1', '6_macro_binary-11', '6_macro_binary-12', '6_macro_binary-2',
        '6_macro_binary-20', '6_macro_binary-21', '6_macro_binary-22', '6_macro_binary-5',
        '6_macro_binary-6', '6_macro_binary-9', '6_macro_cron-24', '6_plugin-26', '6_plugin-28',
        '6_plugin-29', '6_plugin-30', '6_plugin-31', '6_plugin-32', '6_plugin-33', '6_plugin-34',
        '6_plugin-35', '6_screensharing_binary-14', '6_screensharing_binary-39',
        '6_screensharing_binary-4', '6_screensharing_binary-41', '6_screensharing_binary-43',
        '6_screensharing_binary-44', '6_screensharing_binary-45', '6_screensharing_binary-46',
        '6_screensharing_binary-47', '6_screensharing_binary-49', '6_screensharing_binary-51',
        '6_screensharing_binary-53', '6_screensharing_binary-57', '6_screensharing_binary-59',
        '6_screensharing_cron-20', '6_screensharing_cron-21', '6_screensharing_cron-22',
        '6_screensharing_cron-24',
    ],
    'scenario7': [
        '7-10', '7-13', '7-3', '7-6',
    ],
}


def resolve_test_file_arg(value):
    if not value:
        return None
    if value in TEST_SPLIT_PRESETS:
        return TEST_SPLIT_PRESETS[value]
    return [s.strip() for s in value.split(',')]
