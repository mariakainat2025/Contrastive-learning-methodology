"""
TTPDrill sentence filter using exact verb_list.txt and cyber_object_list.txt
from https://github.com/KaiLiu-Leo/TTPDrill-0.5
"""

import os
import re
import sys
import nltk
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

nltk.download('punkt',                      quiet=True)
nltk.download('punkt_tab',                  quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stemmer = PorterStemmer()

# ── Exact TTPDrill verb_list.txt ─────────────────────────────────────────────
_VERB_LIST_RAW = """abusing
accept
accessing
accompany
accomplished
achieves
acquires
acts
activate
adding
adds
adjust
advance
aid
aims
airgapped
allows
allow access
altered
analyze
apis
appears
appends
applying
archived
asking
assemble
assigned
assist
associated
attachments
attacker
attempts
authenticating
auto elevate
auto-elevate
automated
avoids
backdated
based
base32-encoded
base64
base64-encoded
baselined
batch
beacons
becomes
begins
believes
believed
belongs
binds
blend
block
bootkits
breaking
broken
browsing
brute
build
built
bundled
bypassing
bytecode
cached
calls
call back
call-back
callback
capturing
carry
center
chain
changing
checks
check presence
chooses
chosen
circumvent
classify
cleaned
clean up
clears
clicking
clone
closes
cloud
coded
collects
combined
commingling
communicating
compiled
completes
composed
compressing
compromising
computed
concatenating
concealed
conducts
configuring
confuse
connects
consider
consists
contacted
content
continuing
controls
converts
copying
corresponding
cover
cracking
crafte
crawls
creating
customized
data
deactivate
debugging
deceive
decodes
decompresses
decrease
decrypts
defined
degrade
delets
deliever
delivering
demands
deny
deobfuscating
depending
deploys
derived
designed
destroys
detached
detects
deter
determines
developed
directed
disabling
disallow
disconnects
discovers
discovery
disguised
disk
displays
distinguish
distributing
divert
domain
done
double click
double-click
downloads
drives
driven
drops
dumping
duplicates
editing
elapsed
elevating
eliminate
embeds
embedded
employs
emulating
enabling
encapsulating
encoding
encounters
encrypts
end
enforce
engineer
enhance
ensuing
ensuring
enter
enticed
enumerating
escalating
establishing
estimate
evading
evicted
examine
exceed
exchange
execution
exfilling
exfiltrating
exists
exiting
expand
expected
exploits
exported
exposing
extended
extracts
facing
facilitate
fails
faked
falls
featuring
fetch
files
filter
finds
finishes
fixed
flow
flushed
forcing
forged
forms
formatted
forwarding
found
fronting
fulfill
function
gaining
gain access
gain insight
gain-access
gain-insight
gapped
gathers
gave
generating
information
get
grabs
granted
guardrail
guessing
gzipped
handles
hardcoded
harvests
hashed
help
hidden
hiding
highlight
hijacks
hinder
hit
holding
hollows
hooks
hosting
identfy
identifying
image
imitate
impersonating
implements
including
incorporated
increase
indicating
infects
infer
inflate
influenced
inherit
inhibit
initiate
injects
input
inserts
inspecting
installs
instruct
integrated
intended
interacts
intercepting
interfere
interrogate
interrupt
introduced
investigated
invoking
involving
issues
joined
keep
keyboard
kills
knowing
known
lateral movement
launching
lead
leaked
learn
leaving
legitimate
levarage
leveraging
limiting
linked
lists
listens
loads
locates
logs
login
looks
loops
lost
luring
made
maintains
maintain access
making
managing
mangle
manipulating
mapped
marks
masking
masquerading
matching
may
meant
meets
message
migrate
mimics
mimicking
mine
minimize
mislead
mitigate
modification
modifying
modifiy
monitors
mounting
moving
naming
navigate
needed
notified
obfuscates
objects
obscured
observed
obtains
occurs
opens
operating
opposed
ordered
oriented
overwriting
overwritten
overwrote
owned
pass
packs
pad
paid
parses
parsing
passed
password
patches
pay
payload
peer
performs
persists
pertaining
ping
places
planted
plugged
point
poisoned
portopening
possesses
possessed
posting
preceded
predefined
predict
preload
prepares
presents
preserve
pressing
pretend
preventing
print
probing
proceeded
processes
profile
programmed
prohibit
prompts
propagates
protected
providing
proxy
proxy authenticate
proxy execute
pulls
purchased
pushed
puts
querying
ran
ranges
reach
reads
rebooted
receiving
recompile
records
recover
recurring
redirected
reduce
referenced
regained
registers
registry
reinstall
relaunching
relays
relies
reload
remain
remediate
remoting
remote execute
removing
renaming
render
repacked
replacing
replicate
reports
requests
requiring
resembling
resides
resilient
resolve
respond
restart
restore
restricting
resume
retrieving
return
reuse
reveal
reversed
revoked
rewrite
rotating
routes
runs
saving
scans
scheduling
scrapes
scripts
searching
secured
see
seen
segment
selecting
sell
sends
sent
serve
setting
shadow
sharing
shells
shortened
showing
shown
shutting
side
sideload
signs
simulating
skipping
sled
sniffing
sniff
solicit
spawns
spearphising
spearphishing
specializing
specifies
spider
splitting
spoofed
spraying
spreads
stages
starts
stays
steals
stole
stolen
stomped
stops
storing
streams
submitted
subscribe
subverting
suffixed
suggesting
suite
supply
supports
suspended
tables
tags
tailored
taint
taking
taken
targets
terminates
tests
thwart
timestomped
timestop
took
traceroute
tracked
transfers
transmitting
traversing
try
tricking
triggered
trusted
tunneled
turning
undergoes
undermining
understand
undo
undo change
uninstalls
unmap
unmapp
unpacks
updated
upgrade
uploads
uses
using
user
utilizing
vary
vault
verify
versioning
victim
viewing
virtualized
visits
wait
wants
watching
weaponized
what
wiping
wishes
writing
written
zipped"""

# ── Exact TTPDrill cyber_object_list.txt ─────────────────────────────────────
_CYBER_OBJECT_LIST_RAW = """access control
access control list
access tokens
accessibility program
account settings
account username
accounts
active directory ad data
ad infrastructure
ad objects
administrator share
ads
agent
algorithm
alias
anti virus
anti-virus
antivirus
apc queue
apcs
api call
api function call
applescript
application
application vulnerability
application whitelisting
application whitelisting defenses
application whitelisting policies
application whitelisting solutions
applicationsconversations
applicationwhitelisting
applocker
arbitrari script command
arbitrary files
arp
artifacts
ascii
attachment
attributes
audio device
audio recordings
audio tracks
authenticated backdoors
authentication
authentication attempt
authentication mechanisms
authentication values
authoritative source
autorun feature
autostart mechanism
back door
backdoor
banking sites
base64
bashprofile file
bashrc file
behavior
binaries
binary text
bios
bits
bits execution
bits upload functionalities
boot area
boot sectors
bootkits
browser
browser application vulnerability
browser bookmarks
browser functionality
browser security settings
brute force
brute force attacks
brute force login
bug
bypass uac
bypass user account control
bytes
c2 communication
c2 traffic
captured file
cat
cdbexe
cellular phone
certificate
certutil
channel
character
character encoding system
checks
class
clipboard contents
clipboard data
cloned certificate
cloud storage
cmd
cmstpexe
code
code execution
code signing
code signing certificates
code stubs
com object
com scriptlets sct
command
command control
command control c2 communications
command control channel
command control communications
command control protocols
command control server
command control traffic
command line batch script
command line interfaces
command prompt
comment block evasion
common password
common service
common system utility
commonly used ports
communication
communication channels
component object model com
compression library
compromised computer
computer accessories
computer components
computer name
configuration
configuration information
configuration location
connected computer
connection
connection loss
content
content script
control panel
control panel items
cookies
counterfeit product
cpu
credential
credential access
credential access techniques
credential dumping
credentialing mechanisms
credentials keys
custom algorithm
custom command control protocol
custom cryptographic protocol
custom outlook forms
custom tools
data
data execution prevention
data transfer
database
dcom
dde
decryption key
defender
defense
defenses detection analysis
defensive tool
destination
development environment
development tool
device
dga
dictionary
different protocol
digital signatures
directory
distribution mechanisms
dll
dll search order hijacking
dll side loading
dlls
dns
document
domain
domain account
domain admin
domain generation algorithms dgas
domain level group
domain trust relationship
domain trusts
drive size
drivers
drives
dylibs
dynamic data exchange dde execution
elf binary
email address
email data
email file attachment
email payload
email reader
email restriction
embedded command
embedded script
empire
employee
encryption algorithm
end system
endpoint system
entry point binary
environment variable
environmental keying
event
event processes
evidence
executable file
executables
execution control
existing token
exploit code
exploitation
extension
extensions
external hard drive
external remote services
file
file analysis tools
file directory discovery
file extension whitelisting
file extensions
file folder
file hash
file permission
file system
file system permissions weakness
file type
firewall
firewall restrictions
firmware
firmware updates
first bytes
fixed size chunks
folder name
folders drives
forced authentication
forensic investigators
forest environments
ftp
gatekeeper
golden tickets
gpo
gpo modification
gpo settings
gpos
group policy object
guardrails
gzip
hard drives
hardware
hardware architecture
hardware token
hashed credentials
hashes
hidden files
hidden files directories
histcontrol
home page
hooking
hooking mechanisms
host
host information
host locations
hostname
hostname ip address mappings
hta files
html page
http
http management services 80
http sessions
https traffic
hypervisor
ifconfig
ifeo
ifeo mechanisms
image file
images
infected system
information
input capture
input prompt
installed software infromation
integrated camera
integrity security solutions
inter process communication ipc mechanisms
interactive command shells
internet control message protocol icmp
internet web services
interprocess communication ipc
ip address
javascript
jscript
jscript vbscript
kerberoasting
kerberos
kerberos authentication
kerberos ticket
kerberos ticket granting service ticket
kerberos ticket granting ticket
key names
keychain
keylogger
keys
keystroke
language specific semantic
lateral movement
launch agent
launch daemon
ldap
ldapsearch
legitimate program
library
library shared object
link
list
llmnr
lnk directory
lnk file
loadable kernel modules lkms
local account
local file systems
local host file
local system
locality information
location
location executable
log tampering
logical identifier
login
login credentials
login item
logon session
logs
lsa operations
mac address
macro
malware
malware file
masquerading
media firmware
memory
memory addresses
memory legitimate process
memory location
memory process
messages
metasploit
microphones
microsoft office 2016
microsoft office documents
mimikatz
misconfiguration
module
mutex
mysql
name executable
name process
name services
named pipes
netbios
netsh
netstat
network
network activity
network configuration
network connections
network interface
network protocol
network resources
network share connections
network share drive
network shares
network traffic
non standard data encoding system
object
objects schemas
office visual basic applications vba macro
open application
open application windows
open scripting architecture osa language scripts
open source code
open source projects
open windows
operating system
operating system api
operating system api calls
operating system hotfixes
operating system patches
operating system version
outlook
outlook form
outlook rules
packet
passwd
password
password filter dll
password information
password policy
password spraying
password updates
passwords hashes
patches
path
path malicious dll
pathname
payload
peripheral devices
permission
permission groups
permissions setting
persistence
personal information
personal webmail service
php script
physical medium
ping
plain text strings
plaintext credentials
platform
plist
plist file
port
port knock
port redirection
port scans tools
portable executable
powershell
powershell commands
powershell scripts
powersploit
private key
privilege escalation
privilege escalation persistence
privileges
process
process injection
process memory
process monitoring mechanisms
process whitelisting
process whitelisting tool
programming error
programming library
programs
protection mechanisms
protocol traffic
proxies
rainbow table
ransomware
rat
rdp
rdp session hijacking
registry
registry key values
registry keys
remote access software
remote access tools
remote exploitation
remote file copy
remote file shares
remote host service
remote location
remote machines
remote services
remote system information
remote systems
removable media
repositories
requests
resource
restricted information
restrictions
root access
root certificate
rootkit
route
running process
running window
sam
sandboxes
screen
screen capture
screensaver
screensaver setting
screenshot
scripting
scripting language
security context
security mechanisms
security monitoring tools
security policies
security settings
security software
security software discovery
security software information
security support provider
security system
security tools
security vulnerability
security warning
sensitive data
sensor
sensor settings
server information
service binpath
service configuration
service control manager
service executable
service name
service packs
service tickets
service unit files
services
session
session layer protocol
shellcode
shortcut
shortcut modification directory
sid history
sid history injection
signature based detections
signature validation
signature validation restrictions
signature validation tools
size
sleep timers
smb
smb authentication
smtp
social engineering
social media
social media account
software
software packing
software update
software vulnerability
source
source code repository
spearphishing
sqlite3
ssh
ssh connection
ssl client certificates
standard authentication steps
standard data encoding system
standard non application layer protocol
standardized application layer protocol
steganography
structure exception handling
sudoers file configuration
suspended process
system configuration information
system firmware
system firmware update
system functionality
system image
system information
system resources
system setting
system time
system utilities
tainted share content
target path
targets
task scheduling
tasklist
tasklist utility
tcp
template injection
text files
third party software
thread
thread process
time providers
time zone
timestamps file
timestomping
token
token input
token manipulation
token stealing
tool
tools
traffic
traffic pattern
transport layer protocol
trap
trojan
trust controls
trust provider component
trusted directory
uac
unicode
uniform resource locator url
upx
url
usb drive
user account control
user accounts
user activity
user authentication credentials
user datagram protocol
user execution
user input
user rights
username
users
utilities
valid account
vbscript
video
video call service
video clips
video recording
virtual machine environment vme
vulnerability
vulnerability scans tools
weaknesses
web browser
web script
web server
web service
web shell
web sites
webcam
websites
whitelist policy
whitelist process
whitelisting
whitelisting application
whitelisting defenses
whitelisting mechanisms
whitelisting procedures
whitelisting tools
windows api
windows api calls
windows command line utility
windows management instrumentation
windows native api
windows program
windows registry
windows run command
windows service binaries
windows services
windows utilities
wmi
wmi query
wmi scripts
zip
zip file
zlib"""

# Build sets from exact TTPDrill files
ATTACK_VERBS = {
    stemmer.stem(v.strip().split()[0])
    for v in _VERB_LIST_RAW.strip().split('\n')
    if v.strip()
}

CYBER_OBJECTS = [
    obj.strip()
    for obj in _CYBER_OBJECT_LIST_RAW.strip().split('\n')
    if obj.strip()
]

# Build BM25 corpus
_corpus = [obj.lower().split() for obj in CYBER_OBJECTS]
_bm25   = BM25Okapi(_corpus)

BM25_THRESHOLD = 5.0


def _has_attack_verb(sentence):
    words  = re.findall(r'[a-zA-Z]+', sentence.lower())
    stems  = {stemmer.stem(w) for w in words}
    matched = stems & ATTACK_VERBS
    return matched, bool(matched)


def _has_cyber_object(sentence):
    tokens = re.findall(r'[a-zA-Z]+', sentence.lower())
    if not tokens:
        return None, False
    scores    = _bm25.get_scores(tokens)
    best_idx  = scores.argmax()
    best_score = scores[best_idx]
    print(f'         BM25 best score: {best_score:.4f}  matched: {CYBER_OBJECTS[best_idx]}')  # ← add this
    if best_score >= BM25_THRESHOLD:
        return CYBER_OBJECTS[best_idx], True
    return None, False


def filter_sentences(text, verbose=True):
    sentences = nltk.sent_tokenize(text.strip())
    kept    = []
    removed = []
    for sent in sentences:
        verbs, has_verb = _has_attack_verb(sent)
        obj,   has_obj  = _has_cyber_object(sent)
        keep = has_obj
        if verbose and not keep:  # ← only print REMOVE
            print(f'  [REMOVE] {sent.strip()[:90]}')
        if keep:
            kept.append(sent)
        else:
            removed.append(sent)
    return kept, removed


def process_folder(folder_path, verbose=True):
    files = sorted([f for f in os.listdir(folder_path)
                    if f.endswith('.txt') and f != 'benign.txt'])
    results = {}
    for fname in files:
        with open(os.path.join(folder_path, fname)) as f:
            text = f.read().strip()
        print(f'\n{"="*70}\nFILE: {fname}\n{"="*70}')
        kept, removed = filter_sentences(text, verbose=verbose)
        print(f'\n  KEPT: {len(kept)}  REMOVED: {len(removed)}')
        results[fname] = {'kept': kept, 'removed': removed}
    return results


if __name__ == '__main__':
    folder = os.path.join(PROJECT_ROOT, 'input', 'cti_reports')
    process_folder(folder, verbose=True)
