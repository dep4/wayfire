#include <wayfire/idle.hpp>
#include <wayfire/util/log.hpp>
#include "core/seat/input-manager.hpp"
#include "core-impl.hpp"

unsigned int wf::idle_inhibitor_t::inhibitors = 0;

void wf::idle_inhibitor_t::notify_wlroots()
{
    /* NOTE: inhibited -> NOT enabled */
    wlr_idle_notifier_v1_set_inhibited(wf::get_core().protocols.idle_notifier, inhibitors == 0);
    wf::get_core_impl().input->send_inhibit_changed_signal(inhibitors);
}

wf::idle_inhibitor_t::idle_inhibitor_t()
{
    LOGD("creating idle inhibitor ", this, " previous count: ", inhibitors);
    inhibitors++;
    notify_wlroots();
}

wf::idle_inhibitor_t::~idle_inhibitor_t()
{
    LOGD("destroying idle inhibitor ", this, " previous count: ", inhibitors);
    inhibitors--;
    notify_wlroots();
}
