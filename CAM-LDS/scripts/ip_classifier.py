def classify_ip(ip1):
    if (
        ip1 == "0.0.0.0"
        or ip1.startswith("10.")
        or ip1.startswith("127.")
        or ip1.startswith("169.254.")
        or ip1.startswith("192.168.")
        or (
            ip1.startswith("172.")
            and 16 <= int(ip1.split(".")[1]) <= 31
        )
    ):
        return "internal network address"
    else:
        return "external network address"

